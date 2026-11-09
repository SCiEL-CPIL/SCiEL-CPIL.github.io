# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _softplus(x, beta=100.0):
    """Numerically stable softplus."""
    # To avoid overflow in exp for large positive x
    # For large negative x, exp(x) -> 0, log(1) = 0.
    # For large positive x, exp(-x) -> 0, log(1) ~ 0, result ~ x.
    return np.where(x > 20, x, np.log1p(np.exp(x))) / beta

def _compute_penalty(centers, sides, tol=1e-7):
    """
    Compute the penalty for constraint violations.
    Returns a scalar penalty value.
    """
    n = len(sides)
    penalty = 0.0
    
    # 1. Boundary constraints
    # Square i is [cx_i - s_i/2, cx_i + s_i/2] x [cy_i - s_i/2, cy_i + s_i/2]
    # Must be inside [-tol, 1+tol]
    # Violation if left < -tol or right > 1+tol etc.
    # Let's define margin violations.
    # Left margin: cx_i - s_i/2 - (-tol) = cx_i - s_i/2 + tol. Should be >= 0.
    # If negative, penalty.
    
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        
        v_left = -(cx - s/2.0 + tol)     # Positive if violated
        v_right = -(1.0 + tol - cx - s/2.0) # Positive if violated
        v_bottom = -(cy - s/2.0 + tol)
        v_top = -(1.0 + tol - cy - s/2.0)
        
        # Use squared penalty for boundaries to be strong
        # Or softplus. Let's use softplus to keep gradient smooth.
        penalty += _softplus(v_left, beta=50.0)
        penalty += _softplus(v_right, beta=50.0)
        penalty += _softplus(v_bottom, beta=50.0)
        penalty += _softplus(v_top, beta=50.0)
        
    # 2. Overlap constraints
    # For i < j, overlap if |cx_i - cx_j| < (s_i+s_j)/2 - tol AND |cy_i - cy_j| < (s_i+s_j)/2 - tol
    # Let gap_x = |cx_i - cx_j| - (s_i+s_j)/2 + tol. Should be >= 0 OR gap_y >= 0.
    # We want to penalize if BOTH gap_x < 0 AND gap_y < 0.
    # Let A = -gap_x = (s_i+s_j)/2 - tol - |cx_i - cx_j|.
    # Let B = -gap_y = (s_i+s_j)/2 - tol - |cy_i - cy_j|.
    # If A > 0 and B > 0, they overlap.
    # Penalty term: softplus(A) * softplus(B).
    # This is 0 if A is very negative (separated in x) or B is very negative (separated in y).
    # It is large if both are positive.
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            t = (sides[i] + sides[j]) / 2.0 - tol
            
            A = t - dx
            B = t - dy
            
            # Product of softplus
            penalty += _softplus(A, beta=50.0) * _softplus(B, beta=50.0)
            
    return penalty

def _objective(x, n):
    """
    Objective function for minimization.
    We want to maximize sum(sides), so minimize -sum(sides) + Penalty.
    """
    centers = x[:2*n].reshape(n, 2)
    sides = x[2*n:]
    
    # Ensure sides are positive. If optimization pushes them negative, 
    # the boundary penalty will explode, but let's add a small penalty for s<=0 just in case.
    # Actually, bounds in L-BFGS-B handle s > 0.
    
    obj = -np.sum(sides)
    pen = _compute_penalty(centers, sides)
    
    # Weight the penalty heavily to ensure feasibility is prioritized
    # If we don't, it might find a high sum but with overlaps.
    # We want the best *feasible* solution.
    # So we minimize: -sum(sides) + Large_Weight * Penalty
    # If penalty is 0, we maximize sum.
    # If penalty > 0, we reduce it.
    
    return obj + 10.0 * pen

def _unpack(x, n):
    centers = x[:2*n].reshape(n, 2)
    sides = x[2*n:]
    return centers, sides

def _is_feasible(centers, sides, tol=1e-7):
    n = len(sides)
    if not (np.all(np.isfinite(centers)) and np.all(np.isfinite(sides))):
        return False
    if np.any(sides <= 0):
        return False
        
    # Boundaries
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2.0 < -tol: return False
        if cx + s/2.0 > 1.0 + tol: return False
        if cy - s/2.0 < -tol: return False
        if cy + s/2.0 > 1.0 + tol: return False
        
    # Overlaps
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2.0 - tol
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            if dx < t and dy < t:
                return False
    return True

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 580.0  # 20s safety margin
    
    # Initial guess: Grid
    # 10 squares. A 3x3 grid has 9. The 10th can be placed in the center?
    # Or a 2x5 grid?
    # 2x5 grid: side = 1/5 = 0.2. Sum = 10 * 0.2 = 2.0.
    # 3x3 grid (7 squares of 1/3, 3 squares of ?)
    # Let's try a 2x5 grid as a strong baseline.
    # Centers at (0.1, 0.1), (0.3, 0.1)...
    
    # Baseline 1: 2x5 Grid
    k_cols = 5
    k_rows = 2
    side = 1.0 / k_cols
    idx = np.arange(n)
    cols = idx % k_cols
    rows = idx // k_cols
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers0 = np.stack([cx, cy], axis=1).astype(np.float64)
    sides0 = np.full(n, side, dtype=np.float64)
    
    best_centers = centers0.copy()
    best_sides = sides0.copy()
    best_score = np.sum(sides0)
    
    # Bounds for L-BFGS-B
    # cx, cy in [0, 1]
    # sides in [1e-6, 1.0]
    bounds_c = [(0.0, 1.0)] * (2 * n)
    bounds_s = [(1e-6, 1.0)] * n
    bounds = bounds_c + bounds_s
    
    rng = np.random.default_rng(42)
    
    # List of initial configurations
    # 1. Grid
    x0_list = [np.concatenate([centers0.ravel(), sides0.ravel()])]
    
    # 2. Random restarts
    for _ in range(15):
        cx_r = rng.uniform(0.1, 0.9, n)
        cy_r = rng.uniform(0.1, 0.9, n)
        s_r = rng.uniform(0.1, 0.4, n)
        x_r = np.concatenate([np.stack([cx_r, cy_r], axis=1).ravel(), s_r])
        x0_list.append(x_r)
        
    # 3. Shuffled Grid?
    # Just shuffle the grid centers
    perm = rng.permutation(n)
    cx_sh = centers0[perm, 0]
    cy_sh = centers0[perm, 1]
    s_sh = sides0[perm]
    x_sh = np.concatenate([np.stack([cx_sh, cy_sh], axis=1).ravel(), s_sh])
    x0_list.append(x_sh)
    
    # Optimization loop
    for x0 in x0_list:
        if time.time() > deadline:
            break
            
        # Check if this start is already feasible and better?
        c0, s0 = _unpack(x0, n)
        if _is_feasible(c0, s0):
            score0 = np.sum(s0)
            if score0 > best_score:
                best_score = score0
                best_centers = c0
                best_sides = s0
        
        # Run optimization
        try:
            res = minimize(_objective, x0, args=(n,), method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-12, 'maxls': 50})
            
            c_res, s_res = _unpack(res.x, n)
            
            # Check feasibility of the result
            if _is_feasible(c_res, s_res):
                score = np.sum(s_res)
                if score > best_score:
                    best_score = score
                    best_centers = c_res
                    best_sides = s_res
            else:
                # If infeasible, try a quick repair:
                # Shrink sides slightly until feasible?
                # This is a heuristic repair.
                # We can try to reduce all sides by a small factor and check.
                # But this might not be optimal.
                # Let's try a local Nelder-Mead on the penalty only?
                # Or just skip. The penalty method usually converges to feasible.
                # If it doesn't, it's a bad local minimum.
                pass
                
        except Exception as e:
            pass # Ignore errors in optimization

    # Final Check
    if not _is_feasible(best_centers, best_sides):
        # Fallback to the initial grid if everything failed
        return centers0, sides0
        
    return best_centers, best_sides

# EVOLVE-BLOCK-END
