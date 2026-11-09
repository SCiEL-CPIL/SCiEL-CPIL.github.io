# EVOLVE-BLOCK-START
import numpy as np
import time

def _softplus(x, beta=100.0):
    """Numerically stable softplus."""
    return np.where(x > 0, x + np.log1p(np.exp(-x)), np.log1p(np.exp(x))) / beta

def _objective_and_penalty(centers, sides, tol=1e-7):
    """
    Returns (objective, penalty).
    objective = sum(sides)
    penalty = sum of softplus overlap penalties + boundary penalties.
    """
    n = len(sides)
    obj = np.sum(sides)
    
    # Overlap penalty
    pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2.0 - tol
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            # Overlap if dx < t AND dy < t
            # Penalty if t - dx > 0 AND t - dy > 0
            # We want to penalize when both are violated.
            # Using a product of softplus ensures it's 0 if either is satisfied.
            # But product can be tricky for gradients. 
            # Alternative: sum of softplus of (t - dx) and (t - dy) is not enough 
            # because they can overlap in one dimension but not the other.
            # Correct condition: overlap if dx < t AND dy < t.
            # So we need a penalty for the intersection.
            # Let A = t - dx, B = t - dy.
            # If A > 0 and B > 0, there is overlap.
            # Penalty = softplus(A) * softplus(B) works well.
            
            A = t - dx
            B = t - dy
            # If A <= 0 or B <= 0, no overlap.
            # softplus is small for negative inputs.
            # To be safe and smooth, we use:
            # p = softplus(A) * softplus(B)
            # This is 0 if A is very negative or B is very negative.
            # However, if A is slightly positive and B is very negative, product is small.
            # This is good.
            
            # Actually, a simpler and more robust penalty for "AND" condition is:
            # max(0, t-dx) * max(0, t-dy) is non-differentiable.
            # softplus(t-dx) * softplus(t-dy) is differentiable.
            
            pen += _softplus(A) * _softplus(B)
            
    # Boundary penalty
    # Squares must be in [-tol, 1+tol]^2
    # For each square i:
    # left = cx - s/2 >= -tol  => cx - s/2 + tol >= 0
    # right = cx + s/2 <= 1+tol => 1+tol - cx - s/2 >= 0
    # bottom = cy - s/2 >= -tol => cy - s/2 + tol >= 0
    # top = cy + s/2 <= 1+tol => 1+tol - cy - s/2 >= 0
    
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        
        # Penalties for violating boundaries
        # We want to penalize if distance to boundary is negative.
        # d_left = cx - s/2 + tol. If d_left < 0, penalty.
        # d_right = 1 + tol - cx - s/2. If d_right < 0, penalty.
        
        d_left = cx - s/2.0 + tol
        d_right = 1.0 + tol - cx - s/2.0
        d_bottom = cy - s/2.0 + tol
        d_top = 1.0 + tol - cy - s/2.0
        
        pen += _softplus(-d_left)
        pen += _softplus(-d_right)
        pen += _softplus(-d_bottom)
        pen += _softplus(-d_top)
        
    return obj, pen

def _is_feasible(centers, sides, tol=1e-7):
    """Check if the configuration is valid."""
    n = len(sides)
    # Check finite
    if not (np.all(np.isfinite(centers)) and np.all(np.isfinite(sides))):
        return False
    
    # Check side > 0
    if np.any(sides <= 0):
        return False
        
    # Check boundaries
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2.0 < -tol: return False
        if cx + s/2.0 > 1.0 + tol: return False
        if cy - s/2.0 < -tol: return False
        if cy + s/2.0 > 1.0 + tol: return False
        
    # Check overlaps
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
    k = int(np.ceil(np.sqrt(n)))
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers0 = np.stack([cx, cy], axis=1).astype(np.float64)
    sides0 = np.full(n, side, dtype=np.float64)
    
    best_centers = centers0.copy()
    best_sides = sides0.copy()
    best_score = np.sum(sides0)
    
    # Local search using Nelder-Mead
    # We need to optimize 10*2 + 10 = 30 variables.
    # Nelder-Mead is slow for high dimensions, but might work for 30.
    # Alternatively, use L-BFGS-B with penalty.
    
    # Let's try a few random restarts with L-BFGS-B.
    # Variables: x = [cx1, cy1, s1, cx2, cy2, s2, ...]
    
    from scipy.optimize import minimize
    
    def unpack(x):
        c = x[:2*n].reshape(n, 2)
        s = x[2*n:]
        return c, s
    
    def func(x):
        c, s = unpack(x)
        # Ensure sides are positive by using exp? Or just bounds?
        # L-BFGS-B supports bounds.
        # But we need to handle the objective.
        obj, pen = _objective_and_penalty(c, s)
        # We want to maximize obj - pen.
        # So minimize pen - obj.
        return pen - obj
        
    # Bounds:
    # cx, cy in [0, 1] (roughly, since s can be up to 1)
    # Actually cx can be in [0, 1] if s is small. If s=1, cx=0.5.
    # Let's use wide bounds: [0, 1] for centers, [1e-6, 1.0] for sides.
    # Wait, if s=1, cx must be 0.5. The bounds [0,1] for cx are safe.
    # s can be up to 1.
    
    bounds_c = [(0.0, 1.0)] * (2 * n)
    bounds_s = [(1e-6, 1.0)] * n
    bounds = bounds_c + bounds_s
    
    # Try multiple random restarts
    rng = np.random.default_rng(42)
    
    # Number of restarts depends on time
    # Each L-BFGS-B run might take a few seconds.
    # Let's try 10 restarts.
    
    for restart in range(20):
        if time.time() > deadline:
            break
            
        # Random initial point
        # Perturb the grid or random
        if restart == 0:
            x0 = np.concatenate([centers0.ravel(), sides0.ravel()])
        else:
            # Random points in [0.1, 0.9] for centers, [0.1, 0.4] for sides
            cx_rand = rng.uniform(0.1, 0.9, n)
            cy_rand = rng.uniform(0.1, 0.9, n)
            s_rand = rng.uniform(0.1, 0.4, n)
            x0 = np.concatenate([np.stack([cx_rand, cy_rand], axis=1).ravel(), s_rand])
        
        res = minimize(func, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
        
        c_res, s_res = unpack(res.x)
        
        # Check feasibility
        if _is_feasible(c_res, s_res):
            score = np.sum(s_res)
            if score > best_score:
                best_score = score
                best_centers = c_res
                best_sides = s_res
        else:
            # Try to repair? Or just skip.
            # If infeasible, the penalty should have pushed it out, but maybe it didn't.
            # We can try a local Nelder-Mead repair on the best feasible point found so far?
            # For now, skip.
            pass
            
    # Final check
    if not _is_feasible(best_centers, best_sides):
        # Fallback to grid
        return centers0, sides0
        
    return best_centers, best_sides

# EVOLVE-BLOCK-END
