"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import random

def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    """
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return np.sum(r), True

def _grid_init(side):
    """Create a grid of up to N circles."""
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.001
                positions.append([x, y, r])
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.01])
    return np.array(positions[:N])

def _max_objective(params, n):
    """
    Objective: maximize sum(r) - penalty for violations.
    Using max(0, ...) for non-smoothness.
    """
    xy = params[0::3].reshape(-1, 2)
    r = params[1::3].reshape(-1)
    
    # Ensure radii are positive
    r = np.maximum(r, 0.0)
    
    # Wall violations: r > x, r > 1-x, etc.
    wall_viol = 0.0
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        wall_viol += max(0.0, ri - x)
        wall_viol += max(0.0, ri - (1.0 - x))
        wall_viol += max(0.0, ri - y)
        wall_viol += max(0.0, ri - (1.0 - y))
        
    # Overlap violations: r_i + r_j > dist
    overlap_viol = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap_viol += max(0.0, r[i] + r[j] - dist)
            
    penalty = 100.0 * (wall_viol + overlap_viol)
    return -(np.sum(r) - penalty)

def _max_grad(params, n):
    """
    Analytic sub-gradient of the max objective.
    """
    xy = params[0::3].reshape(-1, 2)
    r = params[1::3].reshape(-1)
    
    grad = np.zeros_like(params)
    
    # Gradient of -sum(r) is -1 for r components
    grad[2::3] = -1.0
    
    # Gradient of wall violations
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        
        # max(0, ri - x)
        if ri > x:
            grad[2*i] += 100.0  # d/dri
            grad[3*i] += -100.0 # d/dx
        # max(0, ri - (1-x))
        if ri > 1.0 - x:
            grad[2*i] += 100.0
            grad[3*i] += 100.0
        # max(0, ri - y)
        if ri > y:
            grad[2*i] += 100.0
            grad[3*i+1] += -100.0
        # max(0, ri - (1-y))
        if ri > 1.0 - y:
            grad[2*i] += 100.0
            grad[3*i+1] += 100.0
            
    # Gradient of overlap violations
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            
            if dist == 0:
                # Degenerate case, push apart arbitrarily
                continue
                
            if r[i] + r[j] > dist:
                # Violation exists
                # Term: r_i + r_j - dist
                # d/dri = 1, d/drj = 1
                grad[2*i] += 100.0
                grad[2*j] += 100.0
                
                # d/dxy_i = - (dx_i - dx_j) / dist
                # d/dxy_j = - (dx_j - dx_i) / dist = + (dx_i - dx_j) / dist
                
                ux = dx / dist
                uy = dy / dist
                
                # For circle i (index 3i, 3i+1)
                grad[3*i] += -100.0 * ux
                grad[3*i+1] += -100.0 * uy
                
                # For circle j (index 3j, 3j+1)
                grad[3*j] += 100.0 * ux
                grad[3*j+1] += 100.0 * uy

    return grad

def _rigid_body_optimize(xy, r, deadline):
    """
    Optimize positions and radii using L-BFGS-B with analytic sub-gradients.
    """
    n = len(xy)
    params = np.zeros(3 * n)
    params[0::3] = xy[:, 0]
    params[1::3] = xy[:, 1]
    params[2::3] = r
    
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 0.5))
        
    # Use L-BFGS-B. It handles non-smoothness reasonably well by using history of gradients.
    # Since we provide analytic gradients, it's faster than finite differences.
    
    try:
        result = minimize(
            _max_objective,
            params,
            jac=_max_grad,
            args=(n,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'maxfun': 20000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        opt_params = result.x
        opt_xy = opt_params[0::3].reshape(-1, 2)
        opt_r = np.maximum(opt_params[1::3], 0.0)
        
        # Post-process: ensure validity
        # Clip positions to be inside square given radii
        for i in range(n):
            opt_r[i] = min(opt_r[i], opt_xy[i, 0], 1.0 - opt_xy[i, 0], opt_xy[i, 1], 1.0 - opt_xy[i, 1])
            opt_xy[i, 0] = np.clip(opt_xy[i, 0], opt_r[i], 1.0 - opt_r[i])
            opt_xy[i, 1] = np.clip(opt_xy[i, 1], opt_r[i], 1.0 - opt_r[i])
            
        total, valid = _eval_packing(opt_xy, opt_r)
        return opt_xy, opt_r, total, valid
    except Exception:
        return xy, r, 0.0, False

def _rigid_body_translation(xy, r, deadline, n_attempts=10):
    """
    Try to improve the packing by translating the entire configuration.
    This helps escape local minima where the "center of mass" is stuck.
    """
    n = len(xy)
    best_xy = xy.copy()
    best_r = r.copy()
    best_total, best_valid = _eval_packing(xy, r)
    
    rng = np.random.default_rng(42)
    
    for _ in range(n_attempts):
        if time.time() > deadline:
            break
            
        # Random translation
        tx = rng.uniform(-0.1, 0.1)
        ty = rng.uniform(-0.1, 0.1)
        
        # Random rotation
        theta = rng.uniform(-0.2, 0.2)
        c, s = np.cos(theta), np.sin(theta)
        
        # Center the configuration
        cx = np.mean(xy[:, 0])
        cy = np.mean(xy[:, 1])
        
        new_xy = (xy - np.array([cx, cy])) @ np.array([[c, s], [-s, c]]) + np.array([cx + tx, cy + ty])
        
        # Clip to square
        for i in range(n):
            new_xy[i, 0] = np.clip(new_xy[i, 0], r[i], 1.0 - r[i])
            new_xy[i, 1] = np.clip(new_xy[i, 1], r[i], 1.0 - r[i])
            
        # Optimize the new configuration
        opt_xy, opt_r, total, valid = _rigid_body_optimize(new_xy, r, deadline)
        
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = opt_xy
            best_r = opt_r
            
    return best_xy, best_r, best_total, best_valid

def _bisection_growth(xy, r, deadline, tol=1e-6):
    """
    For each circle, try to increase its radius by binary search,
    adjusting position to maintain validity.
    """
    n = len(xy)
    best_xy = xy.copy()
    best_r = r.copy()
    
    for i in range(n):
        if time.time() > deadline:
            break
            
        # Binary search for max radius of circle i
        lo = best_r[i]
        hi = min(0.5, best_xy[i, 0], 1.0 - best_xy[i, 0], best_xy[i, 1], 1.0 - best_xy[i, 1])
        
        # We need to find a position (x, y) such that circle i with radius R is valid.
        # This is a 2D feasibility problem.
        # Simplification: Assume position stays roughly the same, or do a local search.
        # Let's do a simple local search for the position of circle i given a target radius.
        
        # Actually, let's just try to grow the radius and use _rigid_body_optimize to fix it.
        # This is expensive but effective.
        
        # Try growing by 10%
        new_r = best_r[i] * 1.1
        if new_r > hi:
            new_r = hi
            
        new_xy = best_xy.copy()
        new_r_vec = best_r.copy()
        new_r_vec[i] = new_r
        
        # Optimize
        opt_xy, opt_r_vec, total, valid = _rigid_body_optimize(new_xy, new_r_vec, deadline)
        
        if valid and total > np.sum(best_r):
            best_xy = opt_xy
            best_r = opt_r_vec

    return best_xy, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False
    
    # Initialize with grid
    side = int(np.ceil(np.sqrt(N)))
    xy = _grid_init(side)
    r = xy[:, 2].copy()
    xy = xy[:, :2].copy()
    
    # Step 1: Initial Rigid Body Optimization
    if time.time() < deadline - 30.0:
        out_xy, out_r, total, valid = _rigid_body_optimize(xy, r, deadline - 30.0)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r
    
    # Step 2: Rigid Body Translation Search
    if time.time() < deadline - 10.0 and best_valid:
        out_xy, out_r, total, valid = _rigid_body_translation(best_xy, best_r, deadline - 10.0, n_attempts=5)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r
            
    # Step 3: Bisection Growth (if time permits)
    if time.time() < deadline - 5.0 and best_valid:
        out_xy, out_r = _bisection_growth(best_xy, best_r, deadline - 5.0)
        total, valid = _eval_packing(out_xy, out_r)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r
            
    # Fallback
    if best_xy is None or not best_valid:
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
        return result
    
    # Final validation
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
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
            
    return result
# EVOLVE-BLOCK-END
