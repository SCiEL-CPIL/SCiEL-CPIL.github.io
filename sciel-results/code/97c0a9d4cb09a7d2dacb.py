import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 21

def _perimeter(w, h):
    return 2.0 * (w + h)

def _check_validity(x, y, r, W, H, tol=1e-7):
    """Check if a packing is valid."""
    n = len(x)
    for i in range(n):
        if x[i] < r[i] - tol or x[i] > W - r[i] + tol:
            return False
        if y[i] < r[i] - tol or y[i] > H - r[i] + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True

def _compute_value(x, y, r, W, H):
    """Compute the value (sum of radii) if valid, else -inf."""
    if not _check_validity(x, y, r, W, H):
        return -np.inf
    return np.sum(r)

def _grid_init(W, H, n, seed=0):
    """Create a grid-like initial configuration."""
    rng = np.random.RandomState(seed)
    # Try different grid arrangements
    best_x, best_y, best_r = None, None, None
    best_score = -np.inf
    
    # Possible grid splits for n items
    for rows in range(1, n + 1):
        if n % rows != 0 and rows > 1:
            continue
        cols = n // rows if n % rows == 0 else 0
        if cols == 0:
            continue
        
        # Calculate positions
        x = np.zeros(n)
        y = np.zeros(n)
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                x[idx] = (j + 0.5) * W / cols
                y[idx] = (i + 0.5) * H / rows
        
        # Estimate max radius
        # Check minimum distance between centers
        min_dist = np.inf
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = min(min_dist, dist)
        
        # Also check boundary distances
        for i in range(n):
            min_dist = min(min_dist, x[i], W - x[i], y[i], H - y[i])
        
        r_guess = min_dist / 2.0 * 0.95
        r = np.full(n, r_guess)
        
        # Validate
        if _check_validity(x, y, r, W, H):
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    
    if best_x is not None:
        return best_x, best_y, best_r
    
    # Fallback: random
    x = rng.uniform(0, W, n)
    y = rng.uniform(0, H, n)
    r = np.full(n, 0.05)
    return x, y, r

def _optimize_aspect(W, H, time_budget):
    """Optimize packing for a given aspect ratio."""
    start = time.time()
    deadline = start + time_budget
    
    best_x, best_y, best_r = None, None, None
    best_value = -np.inf
    
    # Initial configurations
    configs = []
    for seed in range(10):
        x, y, r = _grid_init(W, H, N, seed)
        configs.append((x, y, r))
    
    # Also add some random configurations
    rng = np.random.RandomState(42)
    for _ in range(5):
        x = rng.uniform(0, W, N)
        y = rng.uniform(0, H, N)
        r = np.full(N, 0.05)
        configs.append((x, y, r))
    
    # For each config, try to optimize
    for x0, y0, r0 in configs:
        if time.time() > deadline:
            break
        
        # Objective: maximize sum(r) subject to constraints
        # We'll use a penalty approach
        
        def objective(params):
            x = params[:N]
            y = params[N:2*N]
            r = params[2*N:]
            
            penalty = 0.0
            
            # Overlap penalties
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = r[i] + r[j]
                    if dist < min_dist:
                        penalty += (min_dist - dist) ** 2
            
            # Boundary penalties
            for i in range(N):
                if x[i] < r[i]:
                    penalty += (r[i] - x[i]) ** 2
                elif x[i] > W - r[i]:
                    penalty += (x[i] - (W - r[i])) ** 2
                if y[i] < r[i]:
                    penalty += (r[i] - y[i]) ** 2
                elif y[i] > H - r[i]:
                    penalty += (y[i] - (H - r[i])) ** 2
            
            # Maximize sum(r), so minimize -sum(r)
            return 100.0 * penalty - np.sum(r)
        
        x0_full = np.concatenate([x0, y0, r0])
        
        bounds = []
        for i in range(N):
            bounds.append((0, W))
            bounds.append((0, H))
        for i in range(N):
            bounds.append((0, min(W, H) / 2.0))
        
        try:
            res = minimize(objective, x0_full, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.fun < 1e-5:  # Penalty is small
                x_opt = res.x[:N]
                y_opt = res.x[N:2*N]
                r_opt = res.x[2*N:]
                
                if _check_validity(x_opt, y_opt, r_opt, W, H):
                    val = np.sum(r_opt)
                    if val > best_value:
                        best_value = val
                        best_x, best_y, best_r = x_opt.copy(), y_opt.copy(), r_opt.copy()
                        
        except Exception:
            pass
        
        # Also try differential evolution for a subset of configs
        if time.time() < deadline - 30 and x0 is not None:
            try:
                def de_objective(params):
                    x = params[:N]
                    y = params[N:2*N]
                    r = params[2*N:]
                    
                    penalty = 0.0
                    for i in range(N):
                        for j in range(i + 1, N):
                            dx = x[i] - x[j]
                            dy = y[i] - y[j]
                            dist = np.sqrt(dx * dx + dy * dy)
                            min_dist = r[i] + r[j]
                            if dist < min_dist:
                                penalty += (min_dist - dist) ** 2
                    for i in range(N):
                        if x[i] < r[i]:
                            penalty += (r[i] - x[i]) ** 2
                        elif x[i] > W - r[i]:
                            penalty += (x[i] - (W - r[i])) ** 2
                        if y[i] < r[i]:
                            penalty += (r[i] - y[i]) ** 2
                        elif y[i] > H - r[i]:
                            penalty += (y[i] - (H - r[i])) ** 2
                    
                    return 100.0 * penalty - np.sum(r)
                
                bounds_de = []
                for i in range(N):
                    bounds_de.append((0, W))
                    bounds_de.append((0, H))
                for i in range(N):
                    bounds_de.append((0, min(W, H) / 2.0))
                
                res_de = differential_evolution(de_objective, bounds_de, seed=42,
                                                 maxiter=50, tol=1e-8,
                                                 popsize=15, init='latinhypercube')
                
                if res_de.fun < 1e-5:
                    x_opt = res_de.x[:N]
                    y_opt = res_de.x[N:2*N]
                    r_opt = res_de.x[2*N:]
                    
                    if _check_validity(x_opt, y_opt, r_opt, W, H):
                        val = np.sum(r_opt)
                        if val > best_value:
                            best_value = val
                            best_x, best_y, best_r = x_opt.copy(), y_opt.copy(), r_opt.copy()
                            
            except Exception:
                pass
    
    if best_x is None:
        # Fallback
        x, y, r = _grid_init(W, H, N, 0)
        best_x, best_y, best_r = x, y, r
        best_value = np.sum(r)
    
    return best_x, best_y, best_r, best_value

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_value = -np.inf
    best_result = None
    
    # Try different aspect ratios W/H where W + H = 2
    aspect_ratios = [1.0, 1.1, 1.2, 1.3, 1.5, 2.0, 0.8, 0.7]
    
    # Allocate time based on number of aspect ratios
    time_per_aspect = min(10.0, (deadline - start_time) / len(aspect_ratios))
    
    for ratio in aspect_ratios:
        if time.time() > deadline - 5:
            break
        
        H = 2.0 / (1.0 + ratio)
        W = 2.0 - H
        
        # Ensure W, H are positive
        if W <= 0 or H <= 0:
            continue
        
        x, y, r, val = _optimize_aspect(W, H, time_per_aspect)
        
        if val > best_value:
            best_value = val
            best_result = np.column_stack((x, y, r))
    
    if best_result is None:
        # Final fallback
        W, H = 1.0, 1.0
        x, y, r, _ = _grid_init(W, H, N, 0)
        best_result = np.column_stack((x, y, r))
    
    return best_result
