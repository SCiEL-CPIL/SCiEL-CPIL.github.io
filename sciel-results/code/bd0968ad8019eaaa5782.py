import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Initial guess: Jittered grid
    # Try a 5x5 grid layout in a 1x1 square
    # We will place 21 circles.
    
    def run_optimization(initial_w, seed):
        np.random.seed(seed)
        
        # Create a 5x5 grid of centers
        # Spacing 0.2, from 0.1 to 0.9
        xs = np.linspace(0.1, 0.9, 5)
        ys = np.linspace(0.1, 0.9, 5)
        
        # Flatten and take first 21
        grid_xs = []
        grid_ys = []
        for i in range(5):
            for j in range(5):
                if len(grid_xs) < N:
                    grid_xs.append(xs[i])
                    grid_ys.append(ys[j])
        
        grid_xs = np.array(grid_xs)
        grid_ys = np.array(grid_ys)
        
        # Add jitter
        jitter = 0.02
        grid_xs += np.random.uniform(-jitter, jitter, N)
        grid_ys += np.random.uniform(-jitter, jitter, N)
        
        # Initial radii
        r0 = np.full(N, 0.08)
        
        # Variables: x1, y1, r1, ..., xN, yN, rN, W
        x0 = np.concatenate([grid_xs, grid_ys, r0, [initial_w]])
        
        def objective(params):
            if time.time() > deadline:
                return 1e10
            r = params[2*N:3*N]
            return -np.sum(r)
            
        def objective_grad(params):
            # Gradient of -sum(r)
            grad = np.zeros_like(params)
            grad[2*N:3*N] = -1
            return grad

        constraints = []
        
        # Box constraints for W
        # W in [0.5, 1.5]
        
        # Constraints:
        # 1. x_i >= r_i  => x_i - r_i >= 0
        # 2. y_i >= r_i  => y_i - r_i >= 0
        # 3. x_i <= W - r_i => W - x_i - r_i >= 0
        # 4. y_i <= H - r_i => (2-W) - y_i - r_i >= 0
        # 5. dist_ij >= r_i + r_j
        
        # We define constraints as functions returning g(params) >= 0
        
        # Left boundary
        def c_left(params):
            x = params[0:N]
            r = params[2*N:3*N]
            return x - r
        def c_left_grad(params):
            g = np.zeros_like(params)
            g[0:N] = 1
            g[2*N:3*N] = -1
            return g
        constraints.append({'type': 'ineq', 'fun': c_left, 'jac': c_left_grad})
        
        # Bottom boundary
        def c_bottom(params):
            y = params[N:2*N]
            r = params[2*N:3*N]
            return y - r
        def c_bottom_grad(params):
            g = np.zeros_like(params)
            g[N:2*N] = 1
            g[2*N:3*N] = -1
            return g
        constraints.append({'type': 'ineq', 'fun': c_bottom, 'jac': c_bottom_grad})
        
        # Right boundary
        def c_right(params):
            x = params[0:N]
            r = params[2*N:3*N]
            W = params[3*N]
            return W - x - r
        def c_right_grad(params):
            g = np.zeros_like(params)
            g[0:N] = -1
            g[2*N:3*N] = -1
            g[3*N] = 1
            return g
        constraints.append({'type': 'ineq', 'fun': c_right, 'jac': c_right_grad})
        
        # Top boundary
        def c_top(params):
            y = params[N:2*N]
            r = params[2*N:3*N]
            W = params[3*N]
            H = 2 - W
            return H - y - r
        def c_top_grad(params):
            g = np.zeros_like(params)
            g[N:2*N] = -1
            g[2*N:3*N] = -1
            g[3*N] = -1 # d(H-y-r)/dW = d(2-W-y-r)/dW = -1
            return g
        constraints.append({'type': 'ineq', 'fun': c_top, 'jac': c_top_grad})
        
        # Circle-circle constraints
        # For each pair i < j
        pair_indices = []
        for i in range(N):
            for j in range(i+1, N):
                pair_indices.append((i, j))
                
        def c_circles(params):
            x = params[0:N]
            y = params[N:2*N]
            r = params[2*N:3*N]
            vals = []
            for i, j in pair_indices:
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                vals.append(dist - r[i] - r[j])
            return np.array(vals)
            
        def c_circles_grad(params):
            x = params[0:N]
            y = params[N:2*N]
            r = params[2*N:3*N]
            g = np.zeros((len(pair_indices), len(params)))
            for k, (i, j) in enumerate(pair_indices):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-6:
                    dist = 1e-6
                g[k, i] = dx / dist
                g[k, j] = -dx / dist
                g[k, N+i] = dy / dist
                g[k, N+j] = -dy / dist
                g[k, 2*N+i] = -1
                g[k, 2*N+j] = -1
            return g
            
        constraints.append({'type': 'ineq', 'fun': c_circles, 'jac': c_circles_grad})
        
        # Bounds
        # x, y, r >= 0.01
        # W in [0.5, 1.5]
        bounds = []
        for i in range(N):
            bounds.append((0.01, 1.0)) # x
        for i in range(N):
            bounds.append((0.01, 1.0)) # y
        for i in range(N):
            bounds.append((0.01, 0.5)) # r
        bounds.append((0.5, 1.5)) # W
        
        res = minimize(objective, x0, method='SLSQP', jac=objective_grad, 
                       constraints=constraints, bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-8})
        
        return res

    best_score = -np.inf
    best_res = None
    
    # Try a few initial W values and seeds
    for w_init in [1.0, 0.9, 1.1]:
        for seed in [1, 2, 3]:
            if time.time() > deadline:
                break
            try:
                res = run_optimization(w_init, seed)
                if res.success:
                    score = -res.fun
                    if score > best_score:
                        best_score = score
                        best_res = res
            except Exception:
                pass
                
    if best_res is None:
        # Fallback to simple grid
        side = 5
        cells = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count < N:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    r = 0.5 / side - 0.01
                    cells.append([x, y, r])
                    count += 1
        return np.array(cells)
    
    params = best_res.x
    x = params[0:N]
    y = params[N:2*N]
    r = params[2*N:3*N]
    
    # Verify validity
    W = params[3*N]
    H = 2 - W
    
    valid = True
    for i in range(N):
        if x[i] < r[i] - 1e-6 or y[i] < r[i] - 1e-6 or x[i] > W - r[i] + 1e-6 or y[i] > H - r[i] + 1e-6:
            valid = False
            break
    if valid:
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if dist < r[i] + r[j] - 1e-6:
                    valid = False
                    break
            if not valid:
                break
                
    if not valid:
        # Fallback
        side = 5
        cells = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count < N:
                    x = (i + 0.5) / side
                    y = (j + 0.5) / side
                    r = 0.5 / side - 0.01
                    cells.append([x, y, r])
                    count += 1
        return np.array(cells)
        
    return np.column_stack((x, y, r))
