import numpy as np
from scipy.optimize import minimize
import time

def _union_area_at_c(x, y, c):
    """Compute union area of 8x8 axis-aligned rectangles at height c."""
    N = 8
    frac = 1.0 / N
    x0s = x + c * np.arange(N) * frac
    y0s = y + c * np.arange(N) * frac
    
    xs = np.sort(np.concatenate([x0s, x0s + frac]))
    ys = np.sort(np.concatenate([y0s, y0s + frac]))
    
    cx = 0.5 * (xs[:-1] + xs[1:])
    cy = 0.5 * (ys[:-1] + ys[1:])
    dx = np.diff(xs)
    dy = np.diff(ys)
    
    x_cov = (cx[:, None] >= x0s[None, :]) & (cx[:, None] <= x0s[None, :] + frac)
    y_cov = (cy[:, None] >= y0s[None, :]) & (cy[:, None] <= y0s[None, :] + frac)
    
    x_any = x_cov.any(axis=1)
    y_any = y_cov.any(axis=1)
    covered = x_any[:, None] * y_any[None, :]
    
    area = np.sum(covered * dx[:, None] * dy[None, :])
    return area

def _cost_func(params):
    """Average union area over c in [0,1] using 32 samples."""
    x = params[:8]
    y = params[8:]
    c_samples = np.linspace(0, 1, 32)
    total_area = 0.0
    for c in c_samples:
        total_area += _union_area_at_c(x, y, c)
    return total_area / len(c_samples)

def _smooth_cost(params, beta=40.0):
    """Smooth surrogate for cost using soft coverage."""
    x = params[:8]
    y = params[8:]
    N = 8
    frac = 1.0 / N
    c_samples = np.linspace(0, 1, 20)
    
    total_area = 0.0
    for c in c_samples:
        x0s = x + c * np.arange(N) * frac
        y0s = y + c * np.arange(N) * frac
        
        xs = np.sort(np.concatenate([x0s, x0s + frac]))
        ys = np.sort(np.concatenate([y0s, y0s + frac]))
        
        cx = 0.5 * (xs[:-1] + xs[1:])
        cy = 0.5 * (ys[:-1] + ys[1:])
        dx = np.diff(xs)
        dy = np.diff(ys)
        
        # Soft distances
        left_x = cx[:, None] - x0s[None, :]
        right_x = x0s[None, :] + frac - cx[:, None]
        dist_x = np.minimum(left_x, right_x)
        
        left_y = cy[:, None] - y0s[None, :]
        right_y = y0s[None, :] + frac - cy[:, None]
        dist_y = np.minimum(left_y, right_y)
        
        # Soft sigmoid
        bz_x = np.clip(beta * dist_x, -50, 50)
        x_soft = 1.0 / (1.0 + np.exp(-bz_x))
        
        bz_y = np.clip(beta * dist_y, -50, 50)
        y_soft = 1.0 / (1.0 + np.exp(-bz_y))
        
        # Soft max via log-sum-exp
        x_max = np.max(x_soft, axis=1, keepdims=True)
        x_soft_max = np.sum(x_soft / x_max, axis=1) * x_max.flatten()
        
        y_max = np.max(y_soft, axis=1, keepdims=True)
        y_soft_max = np.sum(y_soft / y_max, axis=1) * y_max.flatten()
        
        covered = x_soft_max[:, None] * y_soft_max[None, :]
        area = np.sum(covered * dx[:, None] * dy[None, :])
        total_area += area
    
    return total_area / len(c_samples)

def _smooth_cost_grad(params, beta=40.0):
    """Smooth surrogate for cost with numerical gradient."""
    x = params[:8]
    y = params[8:]
    N = 8
    frac = 1.0 / N
    c_samples = np.linspace(0, 1, 12)
    
    total_area = 0.0
    for c in c_samples:
        x0s = x + c * np.arange(N) * frac
        y0s = y + c * np.arange(N) * frac
        
        xs = np.sort(np.concatenate([x0s, x0s + frac]))
        ys = np.sort(np.concatenate([y0s, y0s + frac]))
        
        cx = 0.5 * (xs[:-1] + xs[1:])
        cy = 0.5 * (ys[:-1] + ys[1:])
        dx = np.diff(xs)
        dy = np.diff(ys)
        
        left_x = cx[:, None] - x0s[None, :]
        right_x = x0s[None, :] + frac - cx[:, None]
        dist_x = np.minimum(left_x, right_x)
        
        left_y = cy[:, None] - y0s[None, :]
        right_y = y0s[None, :] + frac - cy[:, None]
        dist_y = np.minimum(left_y, right_y)
        
        bz_x = np.clip(beta * dist_x, -50, 50)
        x_soft = 1.0 / (1.0 + np.exp(-bz_x))
        
        bz_y = np.clip(beta * dist_y, -50, 50)
        y_soft = 1.0 / (1.0 + np.exp(-bz_y))
        
        x_max = np.max(x_soft, axis=1, keepdims=True)
        x_soft_max = np.sum(x_soft / x_max, axis=1) * x_max.flatten()
        
        y_max = np.max(y_soft, axis=1, keepdims=True)
        y_soft_max = np.sum(y_soft / y_max, axis=1) * y_max.flatten()
        
        covered = x_soft_max[:, None] * y_soft_max[None, :]
        area = np.sum(covered * dx[:, None] * dy[None, :])
        total_area += area
    
    return total_area / len(c_samples)

def _simulated_annealing_step(params, temperature, rng):
    """Propose a new point via Gaussian perturbation scaled by temperature."""
    N = 8
    scale = 0.05 * temperature
    new_params = params + rng.normal(0, scale, size=params.shape)
    return new_params

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    N = 8
    
    def time_ok(limit):
        return (time.time() - start_time) < limit
    
    best_params = None
    best_cost = np.inf
    
    # Phase 1: Multiple restarts with L-BFGS-B using smooth surrogate
    n_restarts = 8
    for restart in range(n_restarts):
        if not time_ok(250):
            break
            
        if restart == 0:
            x0 = np.linspace(0, 1 - 1/N, N)
            y0 = np.linspace(0, 1 - 1/N, N)
        elif restart == 1:
            rng = np.random.default_rng(42)
            x0 = rng.uniform(0, 1 - 1/N, N)
            y0 = rng.uniform(0, 1 - 1/N, N)
        elif restart == 2:
            rng = np.random.default_rng(100)
            x0 = rng.uniform(0.2, 0.8, N)
            y0 = rng.uniform(0.2, 0.8, N)
        elif restart == 3:
            # Concentrated around center
            rng = np.random.default_rng(200)
            x0 = rng.uniform(0.35, 0.65, N)
            y0 = rng.uniform(0.35, 0.65, N)
        elif restart == 4:
            # Spread out
            rng = np.random.default_rng(300)
            x0 = rng.uniform(0, 1, N)
            y0 = rng.uniform(0, 1, N)
        elif restart == 5:
            # Asymmetric
            rng = np.random.default_rng(400)
            x0 = rng.uniform(0, 0.5, N)
            y0 = rng.uniform(0.5, 1, N)
        elif restart == 6:
            # Diagonal pattern
            rng = np.random.default_rng(500)
            x0 = rng.uniform(0.1, 0.9, N)
            y0 = rng.uniform(0.1, 0.9, N)
        else:
            # Tight cluster
            rng = np.random.default_rng(600)
            x0 = rng.uniform(0.4, 0.6, N)
            y0 = rng.uniform(0.4, 0.6, N)
        
        params0 = np.concatenate([x0, y0])
        cost0 = _cost_func(params0)
        
        if cost0 < best_cost:
            best_cost = cost0
            best_params = params0.copy()
        
        # L-BFGS-B optimization with smooth surrogate
        try:
            res = minimize(_smooth_cost_grad, params0, method='L-BFGS-B',
                          options={'maxiter': 1000, 'gtol': 1e-7})
            if res.fun < best_cost or _cost_func(res.x) < best_cost:
                exact_cost = _cost_func(res.x)
                if exact_cost < best_cost:
                    best_cost = exact_cost
                    best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 2: Simulated annealing to escape local minima
    if best_params is not None and time_ok(400):
        rng = np.random.default_rng(12345)
        current_params = best_params.copy()
        current_cost = best_cost
        
        T = 0.1
        T_min = 0.001
        cooling = 0.98
        n_sa_steps = 600
        
        for step in range(n_sa_steps):
            if not time_ok(450):
                break
            
            new_params = _simulated_annealing_step(current_params, T, rng)
            new_cost = _cost_func(new_params)
            
            if new_cost < current_cost:
                current_params = new_params
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_params = current_params.copy()
            else:
                delta = new_cost - current_cost
                if delta < 0.05:
                    prob = np.exp(-delta / max(T, 1e-8))
                    if rng.random() < prob:
                        current_params = new_params
                        current_cost = new_cost
            
            T *= cooling
    
    # Phase 3: Final high-precision polish with L-BFGS-B
    if best_params is not None and time_ok(550):
        try:
            res = minimize(_smooth_cost_grad, best_params, method='L-BFGS-B',
                          options={'maxiter': 2000, 'gtol': 1e-9})
            exact_cost = _cost_func(res.x)
            if exact_cost < best_cost:
                best_cost = exact_cost
                best_params = res.x.copy()
        except Exception:
            pass
        
        # Additional polish with Powell on exact cost
        if time_ok(590):
            try:
                res2 = minimize(_cost_func, best_params, method='Powell',
                               options={'maxiter': 2000, 'xtol': 1e-9, 'ftol': 1e-9})
                if res2.fun < best_cost:
                    best_cost = res2.fun
                    best_params = res2.x.copy()
            except Exception:
                pass
    
    if best_params is None:
        x0 = np.linspace(0, 1 - 1/N, N)
        y0 = np.linspace(0, 1 - 1/N, N)
        best_params = np.concatenate([x0, y0])
    
    x_opt = best_params[:8]
    y_opt = best_params[8:]
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x_opt[i_flat]
    base_y = y_opt[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    # Ensure finite values
    if not np.all(np.isfinite(positions)):
        x0 = np.linspace(0, 1 - 1/N, N)
        y0 = np.linspace(0, 1 - 1/N, N)
        i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        i_flat = i_grid.flatten()
        j_flat = j_grid.flatten()
        base_x = x0[i_flat]
        base_y = y0[j_flat]
        positions = np.stack([base_x, base_y], axis=1)
    
    return positions
