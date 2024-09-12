clear all
close all
clc

%% analysis variable
testname = 'horizon'; % horizon / stateDim / inputDim / perturbation / noise / noiseY
val = 2 ;

% time vector
T = 10; % Number of time steps
dt = 0.1;
t = 0:dt:T;

% default horizon
horizon = 2;

% default A,B
[A,B] = getAB(4); % choose 1/2/3/4, see function at the bottom

% default input
u = ones(1, length(t)+1); % Input 

% default noise
q = [0.0012 0.038 0.0012 0.038]; % weight on state, variance of state noise
Q = diag(q);
R = 0.02; % weight on meassurement, variance of meassurment noise

% default C
C{1} = [1 0 0 0];
C{2} = C{1};
C{3} = [0 0 1 0];
C{4} = C{3};

switch testname
    case 'horizon'
        horizon = val;
    case 'stateDim'
        state_dim = val;
        A = A(1:state_dim, 1:state_dim);
        B = B(1:state_dim);
        q = q(1:state_dim); % weight on state, variance of state noise
        Q = diag(q);
        nx = size(A,1);

        for i = 1:4
            C{i} = eye(nx);
        end

    case 'inputDim'
        input_dim = val;
        u = ones(input_dim, length(t)+1); % Input 
        dim = ones(1, input_dim);
        B = kron(dim, B);
    case 'perturbation'
        perturb = val;
        detlaA = A*perturb;
        A_perturb = A + detlaA;
    case 'noise'
        q = val*q; % scale the default weight
        Q = diag(q);
    case 'noiseY'
        R = val;
end

% save files
%% simulate process
% system dynamics
nx = size(A, 1); % Number of states
nu = size(B, 2); % Number of inputs

% DMHE initial
Pi0 = 100*eye(nx); % Weight matrix for the initial state estimation error (consensus)
M = [1 1 3 3]; % eq.35, also in algorithm 1

% sensors/ meassurements
ny = size(C{1},1);
num_sensor = size(C,2);

% graph
% K = 1;
K = [1/2    0       0     1/2;
     1/2    1/2     0     0;
     0      1/2     1/2   0;
     0      0       1/2   1/2];

num_sensor = size(K,1);

[x_true, x_clean, w_true, y_meas, v_true] = simulate(A,B,C,u,q,R,T,dt,5*ones(nx,1));

% true measurement noise 
for i = 1:num_sensor
    R_true{i} = cov(v_true');
end

%% create regional values
% create regional C matrix for all sensor, eq.4
for i = 1:num_sensor
    % find neighbors
    search_neighbor = K(i,:);
    neighbor = find(search_neighbor);

    a = [];
    for j = 1:length(neighbor)
        c = C{neighbor(j)};
        a = [a, c'];
    end
    C_reg{i} = a';
end

% regional measurement
for i = 1:num_sensor
    % find neighbors
    search_neighbor = K(i,:) ;
    neighbor = find(search_neighbor);

    R_reg{i} = [];
    y_meas_reg{i} = [];
    for j = 1:length(neighbor)
        % regional covariance (R bar), eq.4
        R_reg{i} = blkdiag(R_reg{i}, R_true{i});

        % regional meassurement (y bar), eq.4
        y_meas_reg{i} = [y_meas_reg{i}; y_meas{neighbor(j)}];
    end
end

%% initial phase t = 0-N
if strcmp(testname, 'perturbation')
    A = A_perturb;
end
tic
N = horizon;
% Set initial state
est(:, 1) = zeros(nx,1);
for k = 2:N+1
    % Update state estimates
    w_est(:,k) = sqrt(Q) * randn(nx, 1);
    est(:, k) = est(:, k-1) + dt * est(:, k-1) + w_est(:,k);
end

for i = 1:num_sensor
    sensor = i;
    x_est{sensor} = [est; w_est]; % store estimation
    x0{sensor} = x_est{sensor}; % prior
    Pi{sensor} = Pi0;
    RN{sensor} = kron(eye(N), R_reg{sensor}); % eq.22
    eig_Pi{sensor} = []; % eigen values of Pi
end

%% DMHE
for t = N+1:length(t)
    % calculate Pi (consensus weight)
    for i = 1:num_sensor
        sensor = i;
        [Pi_star, R_star, C_curve] = star_update(C_reg{sensor}, A, N, R_reg{sensor}, RN{sensor}, Q, Pi{sensor}); % eq.21 - 26
        [Pi_tilde{sensor}, ON{sensor}] = ricatti(A, N, C_reg{sensor}, Pi_star, Q, R_star); % eq.27

        % update weight
        RN{sensor} = R_star;
    end

    % local MHE
    for i = 1:num_sensor
       sensor = i;
       Pi{sensor} = Pi_update(M, K, Pi_tilde, Pi{sensor}, sensor);
       eig_Pi{sensor} = [eig_Pi{sensor} eig(Pi{sensor})]; % store Pi eigen value
        
        % Optimization options
        options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp', 'MaxFunctionEvaluations',30000);
        options.MaxFunctionEvaluations = 1e5;

        % find neighbor
        search_neighbor = K(sensor,:);
        neighbor = find(search_neighbor);

        % Initial guess for optimization
        X0 = reshape(x0{sensor}, [], 1); % Flatten to vector form
    
        % Define the cost function handle
        cost_fun = @(X) local_cost(X, C_reg, Q, R_reg, t, nx, Pi, x0, sensor, y_meas_reg, K, neighbor); 
    
        % Define the nonlinear constraints
        nonlcon = @(X) local_constraints(X, A, N, nx);

        % Solve the optimization problem using fmincon (local MHE)
        [X_opt, fval] = fmincon(cost_fun, X0, [], [], [], [], [], [], nonlcon, options);

        % Reshape the optimized result back to matrix form
        X_opt = reshape(X_opt, [], N+1);
    
        % Store the estimated state and noise at current time step
        % x_est{sensor}(:,t-N) = X_opt(:, 1);
        x_est{sensor}(:,t) = X_opt(:,1); % it's better when i use this one
        x0{sensor} = X_opt; % set new estimation as prior
    end
end

%% plot - result
analysis.testname = testname;
analysis.val = val;
analysis.computationTime = [N, toc];

for i = 1:num_sensor
    analysis.mse{i} = rmse(x_est{i}(j,:),x_true(j,:));
end

filename = sprintf('DMHE_%dsensor_%s_%d.mat', num_sensor,testname,val);

save(filename,'analysis','x_est','x_true')

% plot pi eigen values
figure()
hold on
for j = 1:nx
    subplot(nx,1,j)
    hold on
    grid on
    lgd = [];
    for i = 1:num_sensor
        eigen = eig_Pi{i}(j,:);
        plot(eigen, 'LineWidth', 2)
        nextlgd = ['sensor' num2str(i)];
        lgd = [lgd; nextlgd];
    end
    legend (lgd)
    title(['lambda' num2str(j)]);
end
varname = 'eigenval';
saveplot(num_sensor,testname,val,varname)

% plot error state
figure()
for j = 1:nx
    subplot(nx,1,j)
    hold on
    grid on
    lgd = [];
    for i = 1:num_sensor
        error = x_est{i}(j,:) - x_true(j,:); 
        plot(error, 'LineWidth', 2)
        nextlgd = ['sensor' num2str(i)];
        lgd = [lgd; nextlgd];
    end
    legend (lgd)
    title(['e' num2str(j)]);
end
varname = 'error';
saveplot(num_sensor,testname,val,varname)

% plot comparison state
figure()
for j = 1:nx
    subplot(nx,1,j)
    hold on
    grid on
    plot(x_true(j,:),'LineWidth', 2)
    lgd = ['true x '];
    for i = 1:num_sensor
        plot_x_est = x_est{i}(j, :);
        plot(plot_x_est, 'LineWidth', 2)
        nextlgd = ['sensor' num2str(i)];
        lgd = [lgd; nextlgd];
    end
    legend (lgd)
    title(['X' num2str(j)]);
end
varname = 'state';
saveplot(num_sensor,testname,val,varname)

%% function
function Pi = Pi_update(M, K, Pi_tilde, Pi_old, sensor) % eq.31
    % intialize Pi
    Pi = zeros(size(Pi_old));
    % find neighbors
    neighbor = find(K(sensor,:));
    num_neighbor = length(neighbor);

    for i = 1:num_neighbor
        j = neighbor(i);
        % if isempty(Pi_tilde{j})
        %     Pi_tilde{j} = zeros(size(Pi));
        % end
        Pi = Pi + M(i)*K(sensor,j)^2*Pi_tilde{j};
    end
end

function [Pi_tilde, ON] = ricatti(A, N, C_reg, Pi_star, Q, R_star) % eq.27
    ON = []; % eq.5 (use N instead of nx)
    for i = 1:N
        ON  = [ON (C_reg*A^(i-1))'];
    end
    ON = ON';
    
    Pi_tilde = A*Pi_star*A' + Q -A*Pi_star*ON' * (ON*Pi_star*ON' + R_star)^-1 * ON*Pi_star*A'; % eq.27
end

function [Pi_star, R_star, C_curve] = star_update(C_reg, A, N, R_reg, RN, Q, Pi) % eq. 21 - eq.25
    nx = size(A,1);
    sz_Creg = size(C_reg,1);
    
    % eq.21
    C_curve = [];
    for i = 1:N-1
        temp = [];
        temp2 = [];
        for j = 1:N-1-(i-1)
            temp = [temp; C_reg*A^(j-1)];
        end
        row = sz_Creg*i+1 : sz_Creg*N;
        temp2(row,:) = temp;
        C_curve = [C_curve, temp2];
    end

    QN = kron(eye(N-1), Q); % eq.23
    
    R_star = RN + C_curve * QN * C_curve'; % eq.25
    Pi_star = (Pi^-1 + C_reg' * R_reg^-1 * C_reg)^-1; % eq.26
end


%% function optim
function [c, ceq] = local_constraints(X, A, N, nx)    
    % Reshape X to matrix form
    X = reshape(X, [], N+1);

    % Initialize equality constraints
    ceq = [];

    % separate x and w 
    x_est = X(1:nx, :);
    w_est = X(nx+1:nx*2, :);

    % Constraint (7a): State transition
    for n = 1:size(x_est,2)-1
        ceq = [ceq; x_est(:, n+1) - (A * x_est(:, n)) - w_est(:, n)];
    end

    % No inequality constraints
    c = [];
end

function J = local_cost(X, C_reg, Q, R_reg, t, nx, Pi, x0, sensor, y_meas_reg, K, neighbor)
    % Reshape X to matrix form
    X = reshape(X, 2*nx,[]);
    x0{sensor} = reshape(x0{sensor}, 2*nx,[]);

    % Initialize cost
    cost_V = 0;
    cost_W = 0;

    % Measurement noise cost, k = t-N,..., t
    C = C_reg{sensor};
    row = 1:nx;
    y = y_meas_reg{sensor};
    a = [size(y,2)-t+1 size(X,2)];
    for k = 1:min(a)
        y_est = C * X(row, k);
        y_true = y(:,t+(k-1));
        y_true = reshape(y_true, size(y_est));
        cost_V = cost_V + (y_true - y_est)' * R_reg{sensor}^-1 * (y_true - y_est);
    end
    cost_V = 1/2*cost_V;

    % State noise cost, k = t-N, ..., t-1 
    row = nx+1 : nx*2;
    for k = 1:size(X,2)
        cost_W = cost_W + X(row,k)' * Q^-1 * X(row,k);
    end
    cost_W = 1/2*cost_W;

    % prior cost
    X_0 = 0;
    row = 1:nx;
    for j = 1:length(neighbor)
        X_0 = X_0 + K(sensor,neighbor(j))*x0{neighbor(j)}(row, 2); % eq.9
    end
    X_0 = X(row, 1) - X_0; 
    gamma = 0.5 * X_0' * Pi{sensor}^-1 * X_0; % eq. 10

    J = cost_V + cost_W + gamma; % eq.8
end

%% function other
function [A,B] = getAB(dynamics)
    switch dynamics
        case 1 % PAPER DMHE
            A = [0.9962 0.1949  0       0;
                -0.1949 0.3819  0       0;
                 0      0       0       1;
                 0      0       -1.21   1.98];
            B = 0;
    
        case 2 % PAPER LMI
            A = [1  1  0 0;
                 1 -1  0 0;
                 0  0 -1 1;
                 0  0  2 1];
            B = 0;
    
        case 3 % STABLE A
            A = [0.9 0.1 0 0; 
                -0.2 0.8 0 0; 
                 0 0 0.7 0.2; 
                 0 0 -0.1 0.9]; 
            B = 0;
    
        case 4 % STABLE A (WITH INPUT)
            A = [0.9 0.1 0 0; 
                -0.2 0.8 0 0; 
                0 0 0.7 0.2; 
                0 0 -0.1 0.9]; 
            B = [0.1; 
                0.05; 
                0.1; 
                0.05]; % Input matrix
    end 
end

function [x, x_clean, w, y, v] = simulate(A,B,C,u,q,R,T,dt,x0)
% SIMULATE TRUE STATE
    nx = size(A,1);

    % time step
    t = 0:dt:T;

    % state noise
    Qw = diag(q);
    w = sqrt(Qw) * randn(nx, length(t)-1);

    % simulate state
    x(:,1) = x0; % initial
    x_clean(:,1) = x0; % initial
    
    for k = 1:length(t)-1
        x(:, k+1) = A * x(:, k) + B*u(:,k) + w(:,k);
        x_clean(:, k+1) = A * x_clean(:, k) + B*u(:,k);        
    end

% GET MEASUREMENT
    ny = size(C{1},1); % assuming all sensor measure the same number of states
    num_sensor = size(C,2);

    % get measurement output
    v =  sqrt(R) *randn(ny, length(t));
    for i = 1:num_sensor
        y{i} = C{i} * x(:, :) + v(:,k); % result [Y1, Y2, Y3, ...Yt]
    end
end

function saveplot(num_sensor,testname,val,varname)
    fontsize(gcf,scale = 1.5)
    filename = sprintf('DMHE_%dsensor_%s%d_%s.png', num_sensor,testname,val,varname);
    folder_name = 'imgs';
    filepath = fullfile(folder_name, filename);
    set(gcf, 'Position', [50 50 1200 800]);
    saveas(gcf, filepath);
end