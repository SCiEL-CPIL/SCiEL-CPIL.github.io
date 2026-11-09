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
    centers = params[:2*n].reshape(n, 2)
    sides = params[2*n:]
    
    pen = 0.0
    
    # Boundary penalties - vectorized
    left = centers[:, 0] - sides / 2.0
    right = centers[:, 0] + sides / 2.0
    bottom = centers[:, 1] - sides / 2.0
    top = centers[:, 1] + sides / 2.0
    
    pen += 1e6 * np.sum(np.maximum(0, -tol - left))
    pen += 1e6 * np.sum(np.maximum(0, right - (1 + tol)))
    pen += 1e6 * np.sum(np.maximum(0, -tol - bottom))
    pen += 1e6 * np.sum(np.maximum(0, top - (1 + tol)))
    
    # Handle non-positive sides
    neg_sides = sides[sides <= 0]
    if len(neg_sides) > 0:
        pen += 1e6 * np.sum(1.0 - neg_sides)
    
    # Overlap penalties - vectorized using smooth surrogate
    # For each pair (i,j), compute overlap in x and y
    # Use soft-max for smoothness
    eps = 0.005
    total_overlap_pen = 0.0
    
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            t = (sides[i] + sides[j]) / 2.0 - tol
            
            # Smooth max(0, t - dx) and smooth max(0, t - dy)
            vx = t - dx
            vy = t - dy
            
            # soft max(0, x) = eps * log(1 + exp(x/eps))
            svx = eps * np.log(1.0 + np.exp(vx / eps)) if vx > -eps*10 else 0.0
            svy = eps * np.log(1.0 + np.exp(vy / eps)) if vy > -eps*10 else 0.0
            
            total_overlap_pen += 1e5 * svx * svy
    
    pen += total_overlap_pen
    
    return -np.sum(sides) + pen

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
    params_init = np.concatenate([centers.flatten(), sides])
    
    try:
        res = minimize(_objective, params_init, args=(n,), method='Powell',
                       options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
        params_opt = res.x
        c_opt = params_opt[:2*n].reshape(n, 2)
        s_opt = np.maximum(params_opt[2*n:], 1e-6)
        update_best(c_opt, s_opt)
    except Exception:
        pass
    
    # Configuration 2: Different arrangement - 2 large + 8 small
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
        
        params_alt = np.concatenate([centers_alt.flatten(), sides_alt])
        try:
            res = minimize(_objective, params_alt, args=(n,), method='Powell',
                           options={'maxiter': 1500, 'xatol': 1e-9, 'fatol': 1e-9})
            params_opt = res.x
            c_opt = params_opt[:2*n].reshape(n, 2)
            s_opt = np.maximum(params_opt[2*n:], 1e-6)
            update_best(c_opt, s_opt)
        except Exception:
            pass
    
    # Configuration 3: Grid-like 2x5 or 5x2 arrangement
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
        
        params_grid = np.concatenate([centers_grid.flatten(), sides_grid])
        try:
            res = minimize(_objective, params_grid, args=(n,), method='Powell',
                           options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-9})
            params_opt = res.x
            c_opt = params_opt[:2*n].reshape(n, 2)
            s_opt = np.maximum(params_opt[2*n:], 1e-6)
            update_best(c_opt, s_opt)
        except Exception:
            pass
    
    # Multi-start with random perturbations from best
    if best_centers is not None:
        rng = np.random.default_rng(42)
        for restart in range(60):
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
            
            params_pert = np.concatenate([c_pert.flatten(), s_pert])
            
            try:
                res = minimize(_objective, params_pert, args=(n,), method='Powell',
                               options={'maxiter': 600, 'xatol': 1e-9, 'fatol': 1e-9})
                params_opt = res.x
                c_opt = params_opt[:2*n].reshape(n, 2)
                s_opt = np.maximum(params_opt[2*n:], 1e-6)
                update_best(c_opt, s_opt)
            except Exception:
                pass
    
    # Final high-precision polish
    if best_centers is not None and time.time() < deadline - 5:
        params_best = np.concatenate([best_centers.flatten(), best_sides])
        try:
            res = minimize(_objective, params_best, args=(n,), method='Powell',
                           options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12})
            params_opt = res.x
            c_opt = params_opt[:2*n].reshape(n, 2)
            s_opt = np.maximum(params_opt[2*n:], 1e-6)
            update_best(c_opt, s_opt)
        except Exception:
            pass
        
        # Second polish with different method
        if time.time() < deadline - 2:
            try:
                res = minimize(_objective, params_best, args=(n,), method='Nelder-Mead',
                               options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12,
                                        'adaptive': True})
                params_opt = res.x
                c_opt = params_opt[:2*n].reshape(n, 2)
                s_opt = np.maximum(params_opt[2*n:], 1e-6)
                update_best(c_opt, s_opt)
            except Exception:
                pass
    
    if best_centers is None:
        best_centers = centers
        best_sides = sides

    return best_centers, best_sides
