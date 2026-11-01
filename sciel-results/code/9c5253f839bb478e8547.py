import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    best_score = 0.0
    best_params = None
    
    # Softplus function
    def softplus(x):
        return np.log1p(np.exp(x))
    
    # Loss function
    # params: x[0:N], y[0:N], r[0:N], w[N]
    # h = 2 - w
    def loss(params):
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        w = params[3*N]
        h = 2.0 - w
        
        # Ensure w and h are positive
        if w <= 0.1 or h <= 0.1:
            return 1e10
            
        penalty = 0.0
        
        # Boundary penalties
        # x_i - r_i >= 0 -> penalty for r_i - x_i
        penalty += np.sum(softplus(r - x))
        penalty += np.sum(softplus(x + r - w))
        penalty += np.sum(softplus(r - y))
        penalty += np.sum(softplus(y + r - h))
        
        # Overlap penalties
        # dist_ij >= r_i + r_j
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy + 1e-6)
                penalty += softplus(r[i] + r[j] - dist)
                
        # Objective: Maximize sum(r) -> Minimize -sum(r)
        # We want penalty to be 0. 
        # If penalty is non-zero, the config is invalid.
        # We use a large lambda to enforce constraints.
        lambda_pen = 10.0
        return -np.sum(r) + lambda_pen * penalty

    # Initialization strategy
    # Try a few different aspect ratios and grid layouts
    
    num_restarts = 50
    restart_count = 0
    
    while restart_count < num_restarts:
        if time.time() > deadline:
            break
            
        restart_count += 1
        
        # Random aspect ratio w in [0.5, 1.5]
        w_init = 0.5 + np.random.rand() * 1.0
        h_init = 2.0 - w_init
        
        # Initialize positions in a grid
        # Approximate grid size
        side = int(np.ceil(np.sqrt(N)))
        x_init = np.zeros(N)
        y_init = np.zeros(N)
        r_init = np.full(N, 0.05)
        
        for i in range(N):
            ix = i % side
            iy = i // side
            # Center in cell
            x_init[i] = (ix + 0.5) * (w_init / side) + np.random.normal(0, 0.01)
            y_init[i] = (iy + 0.5) * (h_init / side) + np.random.normal(0, 0.01)
            
        params_init = np.concatenate([x_init, y_init, r_init, [w_init]])
        
        # Optimize
        res = minimize(loss, params_init, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-8})
        
        if res.success or res.fun < 1e5:
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:3*N]
            w_opt = res.x[3*N]
            h_opt = 2.0 - w_opt
            
            # Validate
            if w_opt > 0.1 and h_opt > 0.1:
                valid = True
                # Check boundaries
                if np.any(x_opt - r_opt < -1e-5) or np.any(x_opt + r_opt - w_opt > 1e-5):
                    valid = False
                if np.any(y_opt - r_opt < -1e-5) or np.any(y_opt + r_opt - h_opt > 1e-5):
                    valid = False
                # Check overlaps
                if valid:
                    for i in range(N):
                        for j in range(i + 1, N):
                            dist = np.sqrt((x_opt[i]-x_opt[j])**2 + (y_opt[i]-y_opt[j])**2)
                            if dist < r_opt[i] + r_opt[j] - 1e-5:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    score = np.sum(r_opt)
                    if score > best_score:
                        best_score = score
                        best_params = res.x.copy()
                        
    if best_params is not None:
        x = best_params[:N]
        y = best_params[N:2*N]
        r = best_params[2*N:3*N]
        # w is in best_params[3*N], but we only need x,y,r for output
        # The output format is (n, 3): x, y, r
        return np.column_stack((x, y, r))
    else:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        w = 1.0
        h = 1.0
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cx = (i + 0.5) * (w / side)
                cy = (j + 0.5) * (h / side)
                cr = 0.5 / side - 0.01
                cells.append([cx, cy, cr])
        return np.array(cells[:N])
