import numpy as np
from scipy.optimize import minimize
import time

N = 21
DEADLINE = 120.0 # Seconds

def construct():
    start_time = time.time()
    best_score = 0.0
    best_params = None
    
    # Lambda for penalty
    lam = 1000.0
    # Epsilon for smooth max
    eps = 1e-4
    
    def smooth_max(u):
        return (u + np.sqrt(u**2 + eps**2)) / 2.0

    def objective(params):
        w = params[0]
        h = 2.0 - w
        if w <= 0 or h <= 0:
            return 1e10
        
        x = params[1:N+1]
        y = params[N+1:2*N+1]
        r = params[2*N+1:3*N+1]
        
        # Sum of radii
        sum_r = np.sum(r)
        
        # Box violations
        box_viol = 0.0
        for i in range(N):
            # x bounds
            box_viol += smooth_max(r[i] - x[i])**2
            box_viol += smooth_max(w - r[i] - x[i])**2
            # y bounds
            box_viol += smooth_max(r[i] - y[i])**2
            box_viol += smooth_max(h - r[i] - y[i])**2
            
        # Overlap violations
        overlap_viol = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                viol = r[i] + r[j] - dist
                overlap_viol += smooth_max(viol)**2
                
        penalty = lam * (box_viol + overlap_viol)
        return -sum_r + penalty

    # Bounds
    bounds = [(0.2, 1.8)] # w
    for i in range(N):
        bounds.append((0.0, 2.0)) # x
    for i in range(N):
        bounds.append((0.0, 2.0)) # y
    for i in range(N):
        bounds.append((0.0, 1.0)) # r

    # Initial guesses
    # We will try a few different aspect ratios and grid layouts
    num_restarts = 20
    restarts_done = 0
    
    for restart in range(num_restarts):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # Vary initial aspect ratio
        w0 = 1.0 + 0.2 * np.sin(restart * np.pi / 10)
        
        # Initialize grid
        # Use a 5x5 grid, pick 21
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                # Scale to w, h
                x0 = cx * w0
                y0 = cy * (2.0 - w0)
                # Initial radius: half the min dimension of a cell
                # Cell width: w0/side, height: h0/side
                r0 = 0.5 * min(w0, 2.0 - w0) / side
                cells.append([x0, y0, r0])
                
        # Randomly shuffle and take 21
        np.random.shuffle(cells)
        x0 = np.array([c[0] for c in cells[:N]])
        y0 = np.array([c[1] for c in cells[:N]])
        r0 = np.array([c[2] for c in cells[:N]])
        
        # Add small noise
        x0 += np.random.normal(0, 0.01, N)
        y0 += np.random.normal(0, 0.01, N)
        r0 += np.random.normal(0, 0.001, N)
        
        params0 = np.concatenate(([w0], x0, y0, r0))
        
        # Optimize
        try:
            res = minimize(objective, params0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-10})
            params_opt = res.x
            w_opt = params_opt[0]
            h_opt = 2.0 - w_opt
            x_opt = params_opt[1:N+1]
            y_opt = params_opt[N+1:2*N+1]
            r_opt = params_opt[2*N+1:3*N+1]
            
            # Check validity
            valid = True
            # Check box
            for i in range(N):
                if r_opt[i] > x_opt[i] + 1e-6 or r_opt[i] > w_opt - x_opt[i] + 1e-6 or \
                   r_opt[i] > y_opt[i] + 1e-6 or r_opt[i] > h_opt - y_opt[i] + 1e-6:
                    valid = False
                    break
            if valid:
                # Check overlap
                for i in range(N):
                    for j in range(i+1, N):
                        dist = np.sqrt((x_opt[i]-x_opt[j])**2 + (y_opt[i]-y_opt[j])**2)
                        if r_opt[i] + r_opt[j] > dist + 1e-6:
                            valid = False
                            break
                    if not valid: break
            
            if valid:
                score = np.sum(r_opt)
                if score > best_score:
                    best_score = score
                    best_params = (x_opt, y_opt, r_opt)
                    
        except Exception as e:
            pass
        restarts_done += 1

    if best_params is None:
        # Fallback: return a simple valid grid
        # 5x5 grid, 21 circles
        side = 5
        w = 1.0
        h = 1.0
        cells = []
        for i in range(side):
            for j in range(side):
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                x0 = cx * w
                y0 = cy * h
                r0 = 0.5 / side - 1e-3
                cells.append([x0, y0, r0])
        np.random.shuffle(cells)
        x = np.array([c[0] for c in cells[:N]])
        y = np.array([c[1] for c in cells[:N]])
        r = np.array([c[2] for c in cells[:N]])
        return np.column_stack((x, y, r))
        
    x, y, r = best_params
    return np.column_stack((x, y, r))
