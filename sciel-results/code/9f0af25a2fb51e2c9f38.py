import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    start_time = time.time()
    deadline = 140  # 10s margin
    
    best_value = -np.inf
    best_result = None
    
    # Try multiple aspect ratios
    for aspect in [1.0, 0.9, 1.1, 0.8, 1.25, 0.8, 1.2, 1.333, 0.75]:
        if time.time() > deadline:
            break
        
        W = 2.0 * aspect / (1 + aspect)  # W + H = 2, W/H = aspect
        H = 2.0 - W
        
        # Try different grid arrangements
        for m, n in [(5,5), (4,6), (6,4), (3,7), (7,3), (2,11), (11,2)]:
            if time.time() > deadline:
                break
            
            if m * n < N:
                continue
            
            # Initial grid positions
            xs = np.linspace(r_est, W - r_est, m) if m > 1 else np.array([W/2])
            ys = np.linspace(r_est, H - r_est, n) if n > 1 else np.array([H/2])
            
            # Estimate radius
            r_est = min(W/(2*m) if m > 1 else W/2, H/(2*n) if n > 1 else H/2)
            
            # Create grid points
            grid_x, grid_y = np.meshgrid(xs, ys)
            grid_x = grid_x.flatten()
            grid_y = grid_y.flatten()
            
            # Take first N points
            idx = np.random.RandomState(42).permutation(len(grid_x))[:N]
            x0 = grid_x[idx].copy()
            y0 = grid_y[idx].copy()
            r0 = np.full(N, r_est * 0.9)
            
            # Optimize
            result = optimize_packing(x0, y0, r0, W, H)
            if result is not None:
                value = np.sum(result[:, 2])
                if value > best_value:
                    best_value = value
                    best_result = result
    
    if best_result is not None:
        return best_result
    else:
        # Fallback: seed-like solution
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

def optimize_packing(x0, y0, r0, W, H):
    """Optimize circle positions and radii to maximize sum of radii."""
    N = len(x0)
    
    def objective(params):
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        # Check constraints
        # Circles within rectangle
        if np.any(x < r) or np.any(x > W - r) or np.any(y < r) or np.any(y > H - r):
            return 1e6
        if np.any(r <= 0):
            return 1e6
        
        # Non-overlapping
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if dist < r[i] + r[j] - 1e-9:
                    return 1e6
        
        return -np.sum(r)
    
    x0_opt = x0.copy()
    y0_opt = y0.copy()
    r0_opt = r0.copy()
    
    params0 = np.concatenate([x0_opt, y0_opt, r0_opt])
    
    # Use SLSQP
    try:
        res = minimize(objective, params0, method='SLSQP', options={'maxiter': 200, 'ftol': 1e-10})
        if res.success or (res.fun < 1e5):
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:3*N]
            
            # Verify constraints
            valid = True
            if np.any(x_opt < r_opt - 1e-6) or np.any(x_opt > W - r_opt + 1e-6) or \
               np.any(y_opt < r_opt - 1e-6) or np.any(y_opt > H - r_opt + 1e-6):
                valid = False
            if np.any(r_opt <= 0):
                valid = False
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.sqrt((x_opt[i]-x_opt[j])**2 + (y_opt[i]-y_opt[j])**2)
                    if dist < r_opt[i] + r_opt[j] - 1e-6:
                        valid = False
                        break
            
            if valid:
                return np.column_stack([x_opt, y_opt, r_opt])
    except:
        pass
    
    return None
