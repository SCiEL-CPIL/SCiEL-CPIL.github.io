import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_val = -np.inf
    best_result = None
    
    # Variables: x[0..20], y[0..20], r[0..20], W
    # H = 2 - W
    
    def unpack(params):
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        W = params[3*N]
        H = 2.0 - W
        return x, y, r, W, H

    def objective(params):
        x, y, r, W, H = unpack(params)
        
        # Sum of radii
        sum_r = np.sum(r)
        
        # Boundary penalties
        # x_i in [r_i, W-r_i]
        # y_i in [r_i, H-r_i]
        pen_bound = 0.0
        for i in range(N):
            if x[i] < r[i]:
                pen_bound += (r[i] - x[i])**2
            if x[i] > W - r[i]:
                pen_bound += (x[i] - (W - r[i]))**2
            if y[i] < r[i]:
                pen_bound += (r[i] - y[i])**2
            if y[i] > H - r[i]:
                pen_bound += (y[i] - (H - r[i]))**2
        
        # Overlap penalties
        pen_overlap = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                min_d = r[i] + r[j]
                if d < min_d:
                    pen_overlap += (min_d - d)**2
        
        # Regularization to keep W, H positive and reasonable
        # W in [0.5, 1.5] is a good range for perimeter 4
        if W < 0.5 or W > 1.5:
            return 1e10
            
        return -sum_r + 10.0 * (pen_bound + pen_overlap)

    def run_optimization(x0, y0, r0, W0):
        params0 = np.concatenate([x0, y0, r0, [W0]])
        bounds = []
        for i in range(N):
            bounds.append((0.0, 2.0)) # x
        for i in range(N):
            bounds.append((0.0, 2.0)) # y
        for i in range(N):
            bounds.append((1e-5, 1.0)) # r
        bounds.append((0.5, 1.5)) # W
        
        res = minimize(objective, params0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-6})
        
        if not res.success:
            return None, -np.inf
            
        x, y, r, W, H = unpack(res.x)
        
        # Check validity
        valid = True
        for i in range(N):
            if x[i] < r[i] - 1e-8 or x[i] > W - r[i] + 1e-8:
                valid = False
            if y[i] < r[i] - 1e-8 or y[i] > H - r[i] + 1e-8:
                valid = False
            if r[i] <= 0:
                valid = False
                
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    if d < r[i] + r[j] - 1e-8:
                        valid = False
                        break
                if not valid:
                    break
            
        if valid:
            val = np.sum(r)
            return (x, y, r, W, H), val
        else:
            return None, -np.inf

    # Generate initial configurations
    configs = []
    
    # 1. 4x5 Grid
    for W_init in [0.8, 1.0, 1.2]:
        H_init = 2.0 - W_init
        # 4 rows, 5 cols
        rows, cols = 4, 5
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        r0 = np.zeros(N)
        
        # Calculate max radius for this grid
        # width per cell: W/5, height per cell: H/4
        # r = min(W/10, H/8) approx
        r_init = min(W_init / 10.0, H_init / 8.0) * 0.9
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= N: break
                x0[idx] = (j + 0.5) * W_init / cols
                y0[idx] = (i + 0.5) * H_init / rows
                r0[idx] = r_init
                idx += 1
        
        configs.append((x0, y0, r0, W_init))
        
    # 2. 5x4 Grid
    for W_init in [0.8, 1.0, 1.2]:
        H_init = 2.0 - W_init
        rows, cols = 5, 4
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        r0 = np.zeros(N)
        
        r_init = min(W_init / 8.0, H_init / 10.0) * 0.9
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= N: break
                x0[idx] = (j + 0.5) * W_init / cols
                y0[idx] = (i + 0.5) * H_init / rows
                r0[idx] = r_init
                idx += 1
                
        configs.append((x0, y0, r0, W_init))
        
    # 3. 5x5 Grid with 4 removed (center or corners)
    # Let's just use a dense 5x5 and let optimization remove/shrink
    # Actually, we must have exactly 21 circles.
    # So we pick 21 positions from a 5x5 grid.
    # Best to remove corners to allow larger circles in the middle?
    # Or remove from the edge to make it more square?
    
    for W_init in [0.9, 1.0, 1.1]:
        H_init = 2.0 - W_init
        rows, cols = 5, 5
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        r0 = np.zeros(N)
        
        r_init = min(W_init / 10.0, H_init / 10.0) * 0.9
        
        idx = 0
        # Skip corners (0,0), (0,4), (4,0), (4,4)
        for i in range(rows):
            for j in range(cols):
                if (i==0 and j==0) or (i==0 and j==4) or (i==4 and j==0) or (i==4 and j==4):
                    continue
                if idx >= N: break
                x0[idx] = (j + 0.5) * W_init / cols
                y0[idx] = (i + 0.5) * H_init / rows
                r0[idx] = r_init
                idx += 1
                
        configs.append((x0, y0, r0, W_init))

    # Run optimization for each config
    for i, cfg in enumerate(configs):
        if time.time() > deadline:
            break
            
        x0, y0, r0, W0 = cfg
        try:
            res, val = run_optimization(x0, y0, r0, W0)
            if val > best_val:
                best_val = val
                best_result = res
        except Exception as e:
            pass

    # Polish phase: if we have a valid result, try to improve it with tighter tolerances
    if best_result is not None and time.time() < deadline - 10.0:
        x, y, r, W, H = best_result
        params0 = np.concatenate([x, y, r, [W]])
        bounds = []
        for i in range(N):
            bounds.append((0.0, 2.0))
        for i in range(N):
            bounds.append((0.0, 2.0))
        for i in range(N):
            bounds.append((1e-5, 1.0))
        bounds.append((0.5, 1.5))
        
        try:
            res_polish = minimize(objective, params0, method='L-BFGS-B', bounds=bounds, 
                                   options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res_polish.success:
                x_p, y_p, r_p, W_p, H_p = unpack(res_polish.x)
                
                # Check validity
                valid = True
                for i in range(N):
                    if x_p[i] < r_p[i] - 1e-8 or x_p[i] > W_p - r_p[i] + 1e-8:
                        valid = False
                    if y_p[i] < r_p[i] - 1e-8 or y_p[i] > H_p - r_p[i] + 1e-8:
                        valid = False
                    if r_p[i] <= 0:
                        valid = False
                        
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            dx = x_p[i] - x_p[j]
                            dy = y_p[i] - y_p[j]
                            d = np.sqrt(dx*dx + dy*dy)
                            if d < r_p[i] + r_p[j] - 1e-8:
                                valid = False
                                break
                        if not valid:
                            break
                    
                if valid:
                    val_p = np.sum(r_p)
                    if val_p > best_val:
                        best_val = val_p
                        best_result = (x_p, y_p, r_p, W_p, H_p)
        except Exception:
            pass

    if best_result is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x, y, r, W, H = best_result
    # Construct output array (x, y, r)
    # The task says "array of shape (n, 3): n disjoint circles"
    # Usually (x, y, r) is the standard format.
    out = np.column_stack((x, y, r))
    return out
