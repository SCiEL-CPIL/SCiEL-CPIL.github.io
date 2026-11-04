"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    Valid means all circles inside [0,1]^2 and disjoint interiors.
    """
    n = len(xy)
    # Check walls
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    # Check pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return np.sum(r), True

def _smooth_objective(params, n, penalty_weight=100.0):
    """
    Smooth objective function for gradient-based optimization.
    Minimizes: -sum(r) + penalty_weight * (wall_violation + overlap_violation)
    params: flat array of shape (3*n,) containing [x_0, y_0, r_0, x_1, y_1, r_1, ...]
    """
    xy = params[0::3].reshape(-1, 2)
    r = params[1::3].reshape(-1)
    
    # Ensure radii are positive for the penalty calculation, but let optimizer handle bounds
    r_safe = np.maximum(r, 1e-6)
    
    objective = -np.sum(r_safe)
    
    # Wall penalties (smooth)
    # Distance from center to wall must be >= r
    # Violation if r > dist
    # Smooth approximation: max(0, r - dist)^2
    # We use a smooth max: 0.5 * (v + sqrt(v^2 + eps))
    eps = 1e-6
    def smooth_max(v):
        return 0.5 * (v + np.sqrt(v*v + eps))
        
    # Left wall: x - r < 0 -> v = r - x
    v_left = r_safe - xy[:, 0]
    v_right = r_safe - (1.0 - xy[:, 0])
    v_bottom = r_safe - xy[:, 1]
    v_top = r_safe - (1.0 - xy[:, 1])
    
    wall_pen = np.sum(smooth_max(v_left)**2) + np.sum(smooth_max(v_right)**2) + \
               np.sum(smooth_max(v_bottom)**2) + np.sum(smooth_max(v_top)**2)
    
    # Pairwise overlap penalties
    # Distance between centers must be >= r_i + r_j
    # Violation if r_i + r_j > dist
    overlap_pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = r_safe[i] + r_safe[j]
            v = min_dist - dist
            overlap_pen += smooth_max(v)**2
    
    objective += penalty_weight * (wall_pen + overlap_pen)
    return objective

def _gradient_optimize(xy, r, deadline, penalty_weight=100.0):
    """
    Uses L-BFGS-B to optimize positions and radii simultaneously.
    """
    n = len(xy)
    # Flatten params: [x0, y0, r0, x1, y1, r1, ...]
    params = np.zeros(3 * n)
    params[0::3] = xy[:, 0]
    params[1::3] = xy[:, 1]
    params[2::3] = r
    
    # Bounds:
    # x, y in [0, 1]
    # r in [0.001, 0.5] (max possible radius in unit square for one circle is 0.5, but we have 26, so much smaller, but let's be safe)
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.001, 0.5)) # r
    
    start_time = time.time()
    
    # We run minimize in a loop with increasing penalty weight (simulated annealing of penalty)
    # or just a strong penalty. A strong penalty ensures validity but might get stuck in local minima of the penalty.
    # A soft penalty allows some overlap during search to escape local minima.
    
    best_params = params.copy()
    best_obj = _smooth_objective(best_params, n, penalty_weight)
    
    # Try a few penalty weights
    weights = [10.0, 50.0, 200.0, 1000.0]
    
    for w in weights:
        if time.time() > deadline - 5.0:
            break
            
        # Define objective with specific weight
        def obj(p):
            return _smooth_objective(p, n, w)
            
        res = minimize(obj, best_params, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50})
        
        if res.success or res.fun < best_obj:
            # Check if this solution is actually valid (penalty ~ 0)
            xy_new = res.x[0::3].reshape(-1, 2)
            r_new = res.x[1::3].reshape(-1)
            total, valid = _eval_packing(xy_new, r_new)
            
            # We track the best *valid* solution, but also the best *smooth* solution to continue optimization
            # If valid, it's a candidate for final answer
            # If not valid, we still use it to start the next penalty phase if it's better than current best_params in terms of smooth objective
            if valid:
                # This is a good candidate
                pass # We'll return the best valid one at the end
            
            # Update best_params for next iteration if smooth objective is better
            if res.fun < best_obj:
                best_params = res.x.copy()
                best_obj = res.fun
            else:
                # If it didn't improve, stick with previous
                pass
                
    # Extract best valid solution from all runs? 
    # Actually, the loop above only keeps the best smooth solution. 
    # Let's refine: keep track of best valid solution separately.
    
    # Re-run logic to capture best valid
    best_valid_params = None
    best_valid_total = 0.0
    
    for w in weights:
        if time.time() > deadline - 5.0:
            break
            
        def obj(p):
            return _smooth_objective(p, n, w)
            
        # Start from current best_params
        res = minimize(obj, best_params, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50})
        
        if res.fun < best_obj:
            best_params = res.x.copy()
            best_obj = res.fun
            
        # Check validity of res.x
        xy_new = res.x[0::3].reshape(-1, 2)
        r_new = res.x[1::3].reshape(-1)
        total, valid = _eval_packing(xy_new, r_new)
        if valid and total > best_valid_total:
            best_valid_total = total
            best_valid_params = res.x.copy()
            
    if best_valid_params is not None:
        xy_out = best_valid_params[0::3].reshape(-1, 2)
        r_out = best_valid_params[1::3].reshape(-1)
        return xy_out, r_out, best_valid_total, True
    else:
        # Return the best smooth solution, even if invalid, so caller can decide
        xy_out = best_params[0::3].reshape(-1, 2)
        r_out = best_params[1::3].reshape(-1)
        total, valid = _eval_packing(xy_out, r_out)
        return xy_out, r_out, total, valid

def _best_fit_local_search(xy, r, deadline):
    """
    Heuristic local search: for each circle, try to increase radius by moving it to a 'best fit' position.
    This helps escape local minima where gradient gets stuck due to non-differentiable points or tight constraints.
    """
    n = len(xy)
    current_xy = xy.copy()
    current_r = r.copy()
    current_total, current_valid = _eval_packing(current_xy, current_r)
    
    if not current_valid:
        return current_xy, current_r, current_total, current_valid
        
    best_xy = current_xy.copy()
    best_r = current_r.copy()
    best_total = current_total
    
    # Simple local search: try small random moves for each circle
    # This is a fallback for if gradient fails to improve
    for _ in range(50): # 50 iterations
        if time.time() > deadline: break
        improved = False
        for i in range(n):
            # Try 10 random moves
            for _ in range(10):
                old_x, old_y = current_xy[i, 0], current_xy[i, 1]
                new_x = np.clip(old_x + np.random.normal(0, 0.01), current_r[i], 1.0 - current_r[i])
                new_y = np.clip(old_y + np.random.normal(0, 0.01), current_r[i], 1.0 - current_r[i])
                
                # Try to expand radius slightly
                new_r = current_r[i] * 1.005
                
                # Check validity
                old_xy_i = current_xy[i].copy()
                old_r_i = current_r[i]
                
                current_xy[i, 0] = new_x
                current_xy[i, 1] = new_y
                current_r[i] = new_r
                
                total, valid = _eval_packing(current_xy, current_r)
                if valid and total > best_total:
                    best_total = total
                    best_xy = current_xy.copy()
                    best_r = current_r.copy()
                    improved = True
                else:
                    # Revert
                    current_xy[i, 0] = old_x
                    current_xy[i, 1] = old_y
                    current_r[i] = old_r
            
            if time.time() > deadline: break
            
        if not improved:
            break
            
    return best_xy, best_r, best_total, True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False
    
    # Initialization: Grid
    side = int(np.ceil(np.sqrt(N)))
    # 5x5 grid is 25, we need 26. 
    # Let's use a 5x5 grid for first 25, and place the 26th in the middle or a corner gap.
    # Or just use a random init + gradient. Gradient is good at fixing bad inits.
    
    inits = []
    
    # Init 1: Grid
    pos = []
    for i in range(side):
        for j in range(side):
            if len(pos) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.4 / side
                pos.append([x, y, r])
    # Add 26th circle
    pos.append([0.5, 0.5, 0.05]) # Center, small radius
    init1 = np.array(pos[:N])
    inits.append(init1)
    
    # Init 2: Random
    random_xy = np.random.rand(N, 2)
    random_r = np.full(N, 0.05)
    init2 = np.column_stack([random_xy, random_r])
    inits.append(init2)
    
    # Init 3: Random with larger radii, will be fixed by optimizer
    random_xy2 = np.random.rand(N, 2)
    random_r2 = np.full(N, 0.1)
    init3 = np.column_stack([random_xy2, random_r2])
    inits.append(init3)
    
    for init in inits:
        if time.time() > deadline - 30.0: break
        
        xy = init[:, :2].copy()
        r = init[:, 2].copy()
        
        # Ensure initial validity is not strictly required for gradient method, 
        # but starting from a valid state helps. 
        # If invalid, the penalty will push it to valid.
        
        # Run Gradient Optimization
        grad_xy, grad_r, grad_total, grad_valid = _gradient_optimize(xy, r, deadline - 10.0)
        
        if grad_valid and grad_total > best_total:
            best_total = grad_total
            best_valid = True
            best_xy = grad_xy
            best_r = grad_r
            
        # Run Local Search on the result to polish
        if best_valid and time.time() < deadline - 5.0:
            ls_xy, ls_r, ls_total, ls_valid = _best_fit_local_search(best_xy, best_r, deadline - 2.0)
            if ls_valid and ls_total > best_total:
                best_total = ls_total
                best_valid = True
                best_xy = ls_xy
                best_r = ls_r

    # Fallback
    if best_xy is None or not best_valid:
        # Return a valid grid
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
        return result

    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)
    
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
            
    return result
# EVOLVE-BLOCK-END
