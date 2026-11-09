import numpy as np
import time
from scipy.optimize import minimize

def _is_valid(centers, sides, tol=1e-7):
    n = len(sides)
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            t = (sides[i] + sides[j]) / 2.0 - tol
            if dx < t and dy < t:
                return False
    return True

def _objective(params, n=10, tol=1e-7):
    # params: [s_1..s_n, u_1x..u_nx, u_1y..u_ny]
    # s: n
    # u: 2n
    s = params[:n]
    u = params[n:].reshape(n, 2)
    
    # Compute centers from s and u
    # c_x = s/2 + (1-s)*u_x
    # c_y = s/2 + (1-s)*u_y
    c = np.zeros((n, 2))
    c[:, 0] = s/2.0 + (1.0 - s) * u[:, 0]
    c[:, 1] = s/2.0 + (1.0 - s) * u[:, 1]
    
    pen = 0.0
    
    # Overlap penalties
    eps = 0.005
    total_overlap_pen = 0.0
    
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(c[i, 0] - c[j, 0])
            dy = abs(c[i, 1] - c[j, 1])
            t = (s[i] + s[j]) / 2.0 - tol
            
            vx = t - dx
            vy = t - dy
            
            # soft max(0, x)
            # Use a stable log-sum-exp implementation if possible, or just careful calculation
            # If vx is very negative, exp(vx/eps) is 0, log(1)=0.
            # If vx is very positive, exp(vx/eps) is large, log(exp) = vx/eps, result = vx.
            
            # To avoid overflow in exp:
            # max(0, x) approx:
            if vx > 0:
                svx = vx + eps * np.log1p(np.exp(-vx/eps))
            else:
                svx = eps * np.log1p(np.exp(vx/eps))
                
            if vy > 0:
                svy = vy + eps * np.log1p(np.exp(-vy/eps))
            else:
                svy = eps * np.log1p(np.exp(vy/eps))
            
            total_overlap_pen += 1e5 * svx * svy
    
    pen += total_overlap_pen
    
    return -np.sum(s) + pen

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 590  # safety margin
    
    best_centers = None
    best_sides = None
    best_score = -np.inf
    
    def update_best(centers, sides):
        nonlocal best_centers, best_sides, best_score
        if _is_valid(centers, sides):
            score = np.sum(sides)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                best_sides = sides.copy()

    # We need to convert initial centers/sides to params (s, u)
    # u_x = (c_x - s/2) / (1 - s) if s < 1 else 0.5
    # u_y = (c_y - s/2) / (1 - s) if s < 1 else 0.5
    
    def init_to_params(centers, sides):
        s = sides.copy()
        u = np.zeros((n, 2))
        for i in range(n):
            if s[i] < 1.0:
                u[i, 0] = (centers[i, 0] - s[i]/2.0) / (1.0 - s[i])
                u[i, 1] = (centers[i, 1] - s[i]/2.0) / (1.0 - s[i])
            else:
                u[i, 0] = 0.5
                u[i, 1] = 0.5
            # Clip u to [0, 1] to be safe
            u[i, 0] = np.clip(u[i, 0], 0.0, 1.0)
            u[i, 1] = np.clip(u[i, 1], 0.0, 1.0)
        return np.concatenate([s, u.flatten()])

    def params_to_cs(params):
        s = params[:n]
        u = params[n:].reshape(n, 2)
        c = np.zeros((n, 2))
        c[:, 0] = s/2.0 + (1.0 - s) * u[:, 0]
        c[:, 1] = s/2.0 + (1.0 - s) * u[:, 1]
        return c, s

    # Configuration 1: 4 large at corners + 6 small
    s_large = 0.375
    s_small = 0.25
    
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    
    # 4 Large at corners
    centers[0] = [s_large/2, s_large/2]
    sides[0] = s_large
    centers[1] = [1 - s_large/2, s_large/2]
    sides[1] = s_large
    centers[2] = [s_large/2, 1 - s_large/2]
    sides[2] = s_large
    centers[3] = [1 - s_large/2, 1 - s_large/2]
    sides[3] = s_large
    
    # 4 Small in central strip
    centers[4] = [0.5, 0.125]
    sides[4] = s_small
    centers[5] = [0.5, 0.375]
    sides[5] = s_small
    centers[6] = [0.5, 0.625]
    sides[6] = s_small
    centers[7] = [0.5, 0.875]
    sides[7] = s_small
    
    # 2 Small in side gaps
    centers[8] = [0.1875, 0.5]
    sides[8] = s_small
    centers[9] = [0.8125, 0.5]
    sides[9] = s_small
    
    update_best(centers, sides)
    
    # Optimization from config 1
    params_init = init_to_params(centers, sides)
    
    # Bounds for L-BFGS-B
    # s: [1e-6, 1]
    # u: [0, 1]
    bounds = [(1e-6, 1.0)] * n + [(0.0, 1.0)] * (2 * n)
    
    try:
        res = minimize(_objective, params_init, args=(n,), method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        params_opt = res.x
        c_opt, s_opt = params_to_cs(params_opt)
        update_best(c_opt, s_opt)
    except Exception:
        pass
    
    # Configuration 2: Different arrangement
    if time.time() < deadline - 30:
        s_large2 = 0.42
        s_small2 = 0.23
        
        centers_alt = np.zeros((n, 2))
        sides_alt = np.zeros(n)
        
        # 2 large squares at diagonal corners
        centers_alt[0] = [s_large2/2, s_large2/2]
        sides_alt[0] = s_large2
        centers_alt[1] = [1 - s_large2/2, 1 - s_large2/2]
        sides_alt[1] = s_large2
        
        # 8 small squares
        # Top-left area
        centers_alt[2] = [1 - s_large2/2 - s_small2/2, s_large2/2 + s_small2/2]
        sides_alt[2] = s_small2
        # Top-right area  
        centers_alt[3] = [1 - s_large2/2 - s_small2/2, 1 - s_large2/2 - s_small2/2]
        sides_alt[3] = s_small2
        # Bottom-left area
        centers_alt[4] = [s_large2/2 + s_small2/2, s_large2/2 + s_small2/2]
        sides_alt[4] = s_small2
        # Bottom-right area
        centers_alt[5] = [s_large2/2 + s_small2/2, 1 - s_large2/2 - s_small2/2]
        sides_alt[5] = s_small2
        
        # Center column
        centers_alt[6] = [0.5, 0.25]
        sides_alt[6] = s_small2
        centers_alt[7] = [0.5, 0.5]
        sides_alt[7] = s_small2
        centers_alt[8] = [0.5, 0.75]
        sides_alt[8] = s_small2
        centers_alt[9] = [0.5, 0.5]
        sides_alt[9] = s_small2
        
        update_best(centers_alt, sides_alt)
        
        params_alt = init_to_params(centers_alt, sides_alt)
        try:
            res = minimize(_objective, params_alt, args=(n,), method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            params_opt = res.x
            c_opt, s_opt = params_to_cs(params_opt)
            update_best(c_opt, s_opt)
        except Exception:
            pass
    
    # Configuration 3: Grid-like
    if time.time() < deadline - 30:
        s_grid = 0.2
        centers_grid = np.zeros((n, 2))
        sides_grid = np.zeros(n)
        
        # 5 columns, 2 rows
        for i in range(10):
            row = i // 5
            col = i % 5
            cx = s_grid/2 + col * s_grid
            cy = s_grid/2 + row * s_grid
            centers_grid[i] = [cx, cy]
            sides_grid[i] = s_grid
        
        update_best(centers_grid, sides_grid)
        
        params_grid = init_to_params(centers_grid, sides_grid)
        try:
            res = minimize(_objective, params_grid, args=(n,), method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            params_opt = res.x
            c_opt, s_opt = params_to_cs(params_opt)
            update_best(c_opt, s_opt)
        except Exception:
            pass
    
    # Multi-start with random perturbations from best
    if best_centers is not None:
        rng = np.random.default_rng(42)
        for restart in range(80):
            if time.time() > deadline - 15:
                break
            
            c_pert = best_centers.copy()
            s_pert = best_sides.copy()
            
            c_pert += rng.normal(0, 0.03, size=c_pert.shape)
            s_pert += rng.normal(0, 0.01, size=s_pert.shape)
            s_pert = np.maximum(s_pert, 0.05)
            
            for i in range(n):
                s = s_pert[i]
                c_pert[i, 0] = np.clip(c_pert[i, 0], s/2, 1 - s/2)
                c_pert[i, 1] = np.clip(c_pert[i, 1], s/2, 1 - s/2)
            
            params_pert = init_to_params(c_pert, s_pert)
            
            try:
                res = minimize(_objective, params_pert, args=(n,), method='L-BFGS-B',
                               bounds=bounds, options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
                params_opt = res.x
                c_opt, s_opt = params_to_cs(params_opt)
                update_best(c_opt, s_opt)
            except Exception:
                pass
    
    # Final high-precision polish
    if best_centers is not None and time.time() < deadline - 5:
        params_best = init_to_params(best_centers, best_sides)
        try:
            res = minimize(_objective, params_best, args=(n,), method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10})
            params_opt = res.x
            c_opt, s_opt = params_to_cs(params_opt)
            update_best(c_opt, s_opt)
        except Exception:
            pass
        
        # Second polish with Nelder-Mead on original variables? 
        # Or just another L-BFGS-B with different settings?
        if time.time() < deadline - 2:
            try:
                # Try Powell on original variables for a different perspective
                params_orig = np.concatenate([best_centers.flatten(), best_sides])
                # We need an objective for original variables to use Powell
                # Re-use _objective? No, _objective expects new params.
                # We can create a wrapper or just skip if L-BFGS-B is good enough.
                # Let's try to run Powell on the original params using a wrapper that converts to new params?
                # That's complex. Let's just run L-BFGS-B again with a tighter tol.
                res = minimize(_objective, params_best, args=(n,), method='L-BFGS-B',
                               bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12})
                params_opt = res.x
                c_opt, s_opt = params_to_cs(params_opt)
                update_best(c_opt, s_opt)
            except Exception:
                pass
        
    if best_centers is None:
        best_centers = centers
        best_sides = sides

    return best_centers, best_sides
