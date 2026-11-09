import numpy as np
import scipy.optimize
import time

def _is_valid_vectorized(centers, sides, tol=1e-7):
    n = len(sides)
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
        
    # Boundary check
    if np.any(centers[:, 0] - sides/2 < -tol): return False
    if np.any(centers[:, 0] + sides/2 > 1 + tol): return False
    if np.any(centers[:, 1] - sides/2 < -tol): return False
    if np.any(centers[:, 1] + sides/2 > 1 + tol): return False
    
    # Overlap check
    dx = np.abs(centers[:, 0][:, np.newaxis] - centers[:, 0][np.newaxis, :])
    dy = np.abs(centers[:, 1][:, np.newaxis] - centers[:, 1][np.newaxis, :])
    sep = (sides[:, np.newaxis] + sides[np.newaxis, :]) / 2.0 - tol
    mask = ~np.eye(n, dtype=bool)
    
    if np.any((dx < sep) & (dy < sep) & mask):
        return False
        
    return True

def _penalty_objective(x, n=10, tol=1e-7):
    centers = x[:2*n].reshape(n, 2)
    sides = x[2*n:]
    
    obj = -np.sum(sides)
    
    # Boundary penalties
    left_viol = np.maximum(0, -(centers[:, 0] - sides/2))
    right_viol = np.maximum(0, (centers[:, 0] + sides/2 - 1))
    bottom_viol = np.maximum(0, -(centers[:, 1] - sides/2))
    top_viol = np.maximum(0, (centers[:, 1] + sides/2 - 1))
    
    bound_pen = np.sum(left_viol**2) + np.sum(right_viol**2) + \
                np.sum(bottom_viol**2) + np.sum(top_viol**2)
    
    # Overlap penalties
    dx = np.abs(centers[:, 0][:, np.newaxis] - centers[:, 0][np.newaxis, :])
    dy = np.abs(centers[:, 1][:, np.newaxis] - centers[:, 1][np.newaxis, :])
    sep_req = (sides[:, np.newaxis] + sides[np.newaxis, :]) / 2.0 - tol
    
    pen_x = np.maximum(0, sep_req - dx)
    pen_y = np.maximum(0, sep_req - dy)
    
    i_idx, j_idx = np.triu_indices(n, 1)
    pair_pen = pen_x[i_idx, j_idx] * pen_y[i_idx, j_idx]
    overlap_pen = np.sum(pair_pen**2)
    
    neg_side_pen = np.sum(np.maximum(0, -sides)**2)
    
    weight = 100.0 # Strong penalty
    
    return obj + weight * (bound_pen + overlap_pen + neg_side_pen)

def _polish_solution(centers, sides, max_time=60):
    n = len(sides)
    x0 = np.concatenate([centers.flatten(), sides])
    
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((0.01, 1.0))
        
    start_time = time.time()
    
    try:
        res = scipy.optimize.minimize(
            _penalty_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-10, 'maxfun': 500}
        )
        
        if time.time() - start_time > max_time:
            return None
            
        x_opt = res.x
        new_centers = x_opt[:2*n].reshape(n, 2)
        new_sides = x_opt[2*n:]
        
        if _is_valid_vectorized(new_centers, new_sides, tol=1e-7):
            return new_centers, new_sides
        else:
            # Clip and re-check
            for i in range(n):
                s = new_sides[i]
                new_centers[i, 0] = np.clip(new_centers[i, 0], s/2, 1 - s/2)
                new_centers[i, 1] = np.clip(new_centers[i, 1], s/2, 1 - s/2)
            if _is_valid_vectorized(new_centers, new_sides, tol=1e-7):
                return new_centers, new_sides
            else:
                return None
    except Exception:
        return None

def squares_in_square10() -> tuple:
    n = 10
    tol = 1e-7
    start_time = time.time()
    deadline = 600 # Safety margin from 630
    
    best_centers = None
    best_sides = None
    best_val = -np.inf
    
    # Initial known good packing (Sum 3.0)
    centers_init = np.zeros((n, 2))
    sides_init = np.zeros(n)
    centers_init[0] = [0.25, 0.25]; sides_init[0] = 0.5
    centers_init[1] = [0.75, 0.25]; sides_init[1] = 0.5
    for k in range(8):
        row = k // 4
        col = k % 4
        cy = 0.625 + row * 0.25
        cx = 0.125 + col * 0.25
        centers_init[2+k] = [cx, cy]
        sides_init[2+k] = 0.25
        
    if _is_valid_vectorized(centers_init, sides_init, tol):
        best_centers = centers_init.copy()
        best_sides = sides_init.copy()
        best_val = np.sum(sides_init)
        
    # Differential Evolution
    # Bounds: cx, cy in [0,1], s in [0.01, 1]
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.01, 1.0)] * n
    
    # We will run DE for a few iterations
    # DE is slow, so we limit maxiter and popsize
    # We will run multiple DE instances with different seeds
    
    seeds = [42, 123, 999]
    
    for seed in seeds:
        if time.time() > deadline - 120: # Leave time for polish
            break
            
        try:
            # DE parameters
            # maxiter: 100 is usually enough to get close
            # popsize: 15
            de_res = scipy.optimize.differential_evolution(
                _penalty_objective,
                bounds=bounds,
                args=(n,),
                seed=seed,
                maxiter=100,
                popsize=15,
                tol=1e-5,
                mutation=(0.5, 1.5),
                recombination=0.9,
                polish=False # We will polish manually
            )
            
            x_opt = de_res.x
            new_centers = x_opt[:2*n].reshape(n, 2)
            new_sides = x_opt[2*n:]
            
            # Check validity of DE result
            if _is_valid_vectorized(new_centers, new_sides, tol):
                val = np.sum(new_sides)
                if val > best_val:
                    best_val = val
                    best_centers = new_centers.copy()
                    best_sides = new_sides.copy()
            else:
                # Try to repair
                for i in range(n):
                    s = new_sides[i]
                    new_centers[i, 0] = np.clip(new_centers[i, 0], s/2, 1 - s/2)
                    new_centers[i, 1] = np.clip(new_centers[i, 1], s/2, 1 - s/2)
                if _is_valid_vectorized(new_centers, new_sides, tol):
                    val = np.sum(new_sides)
                    if val > best_val:
                        best_val = val
                        best_centers = new_centers.copy()
                        best_sides = new_sides.copy()
                        
        except Exception:
            pass
            
    # Polish the best solution
    if best_centers is not None and time.time() < deadline - 10:
        polished = _polish_solution(best_centers, best_sides, max_time=deadline - time.time() - 5)
        if polished is not None:
            p_centers, p_sides = polished
            p_val = np.sum(p_sides)
            if p_val > best_val:
                best_val = p_val
                best_centers = p_centers
                best_sides = p_sides
                
    return best_centers, best_sides
