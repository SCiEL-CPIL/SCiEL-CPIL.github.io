import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _objective(params, N):
    w = params[0]
    h = 2.0 - w
    if w <= 0.01 or h <= 0.01:
        return 1e10
    
    # Extract circle parameters
    xs = params[1:N+1]
    ys = params[N+1:2*N+1]
    rs = params[2*N+1:3*N+1]
    
    # Sum of radii
    sum_r = np.sum(rs)
    
    # Inside rectangle penalty
    penalty = 0.0
    for i in range(N):
        r = rs[i]
        x = xs[i]
        y = ys[i]
        p_left = max(0.0, r - x)
        p_right = max(0.0, r - (w - x))
        p_bottom = max(0.0, r - y)
        p_top = max(0.0, r - (h - y))
        penalty += p_left**2 + p_right**2 + p_bottom**2 + p_top**2
    
    # Disjointness penalty
    for i in range(N):
        for j in range(i+1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            overlap = max(0.0, rs[i] + rs[j] - dist)
            penalty += overlap**2
    
    lambda_pen = 500.0
    return -sum_r + lambda_pen * penalty

def _is_valid(params, N, tol=1e-6):
    w = params[0]
    h = 2.0 - w
    if w <= 0 or h <= 0:
        return False
    
    xs = params[1:N+1]
    ys = params[N+1:2*N+1]
    rs = params[2*N+1:3*N+1]
    
    for i in range(N):
        r = rs[i]
        x = xs[i]
        y = ys[i]
        if r <= 0:
            return False
        if x - r < -tol or w - x - r < -tol or y - r < -tol or h - y - r < -tol:
            return False
        for j in range(i+1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < rs[i] + rs[j] - tol:
                return False
    return True

def _make_grid_init(w, h, ncols, nrows):
    # Place circles in a grid
    params = np.zeros(3*N + 1)
    params[0] = w
    
    count = 0
    for i in range(ncols):
        for j in range(nrows):
            if count >= N:
                break
            x = (i + 0.5) * w / ncols
            y = (j + 0.5) * h / nrows
            r = min(w / ncols, h / nrows) / 2 * 0.95
            params[1 + count] = x
            params[N + 1 + count] = y
            params[2*N + 1 + count] = r
            count += 1
    
    # Fill remaining with small circles at random positions
    for count in range(N):
        if params[1+count] == 0 and params[N+1+count] == 0 and params[2*N+1+count] == 0:
            params[1+count] = np.random.uniform(0.1*w, 0.9*w)
            params[N+1+count] = np.random.uniform(0.1*h, 0.9*h)
            params[2*N+1+count] = 0.01
    
    return params

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_value = 0.0
    best_params = None
    
    # Try different aspect ratios and grid configurations
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5]
    grid_configs = [(5,5), (4,6), (6,4), (7,3), (3,7), (4,5), (5,4)]
    
    rng = np.random.RandomState(42)
    
    attempts = 0
    for ar in aspect_ratios:
        if time.time() > deadline - 15:
            break
        # w/h = ar, w + h = 2 => w = 2*ar/(1+ar), h = 2/(1+ar)
        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 / (1.0 + ar)
        
        for (ncols, nrows) in grid_configs:
            if time.time() > deadline - 15:
                break
            params0 = _make_grid_init(w, h, ncols, nrows)
            result = minimize(_objective, params0, args=(N,), method='L-BFGS-B',
                            bounds=[(0.01, 1.99)] + [(0.01, 1.99)]*(N) + [(0.01, 1.99)]*(N) + [(0.001, 1.0)]*(N),
                            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if result.success or result.fun < 0:
                if _is_valid(result.x, N):
                    val = np.sum(result.x[2*N+1:3*N+1])
                    if val > best_value:
                        best_value = val
                        best_params = result.x.copy()
            
            attempts += 1
    
    # Random starts
    for _ in range(10):
        if time.time() > deadline - 10:
            break
        w = rng.uniform(0.5, 1.5)
        h = 2.0 - w
        params0 = np.zeros(3*N + 1)
        params0[0] = w
        for i in range(N):
            params0[1+i] = rng.uniform(0.1*w, 0.9*w)
            params0[N+1+i] = rng.uniform(0.1*h, 0.9*h)
            params0[2*N+1+i] = rng.uniform(0.01, 0.05)
        
        result = minimize(_objective, params0, args=(N,), method='L-BFGS-B',
                        bounds=[(0.01, 1.99)] + [(0.01, 1.99)]*(N) + [(0.01, 1.99)]*(N) + [(0.001, 1.0)]*(N),
                        options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if result.success or result.fun < 0:
            if _is_valid(result.x, N):
                val = np.sum(result.x[2*N+1:3*N+1])
                if val > best_value:
                    best_value = val
                    best_params = result.x.copy()
    
    # If we found a valid solution, return it
    if best_params is not None:
        w = best_params[0]
        h = 2.0 - w
        result = np.zeros((N, 3))
        for i in range(N):
            result[i] = [best_params[1+i], best_params[N+1+i], best_params[2*N+1+i]]
        return result
    
    # Fallback: seed solution
    side = 5
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
