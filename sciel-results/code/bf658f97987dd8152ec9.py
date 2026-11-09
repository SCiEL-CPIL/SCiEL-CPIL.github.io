import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _is_feasible(centers, sides, tol=1e-7):
    """Check if the configuration is feasible."""
    n = len(sides)
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if s <= 0:
            return False
        if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(s)):
            return False
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j])/2 - tol
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            if dx < t and dy < t:
                return False
    return True

def _make_2big_8small():
    """Construct 2 large (0.5) + 8 small (0.25) tiling. Sum = 3.0."""
    centers = np.array([
        [0.25, 0.75],
        [0.75, 0.75],
        [0.125, 0.125], [0.375, 0.125], [0.625, 0.125], [0.875, 0.125],
        [0.125, 0.375], [0.375, 0.375], [0.625, 0.375], [0.875, 0.375]
    ])
    sides = np.array([0.5, 0.5] + [0.25]*8)
    return centers, sides

def _smooth_penalty(x, n=10, k=100.0):
    """
    Calculate smooth penalty for overlaps and boundary violations.
    x: array of length 3*n [cx_0..cx_n, cy_0..cy_n, s_0..s_n]
    """
    centers = x[:2*n].reshape(n, 2)
    sides = x[2*n:]
    
    penalty = 0.0
    
    # Boundary violations
    for i in range(n):
        cx, cy, s = centers[i, 0], centers[i, 1], sides[i]
        # Lower bound
        if cx - s/2 < 0:
            penalty += (0 - (cx - s/2))**2
        if cy - s/2 < 0:
            penalty += (0 - (cy - s/2))**2
        # Upper bound
        if cx + s/2 > 1:
            penalty += ((cx + s/2) - 1)**2
        if cy + s/2 > 1:
            penalty += ((cy + s/2) - 1)**2
    
    # Overlap violations
    for i in range(n):
        for j in range(i+1, n):
            cx_i, cy_i = centers[i]
            cx_j, cy_j = centers[j]
            s_i, s_j = sides[i], sides[j]
            
            t_x = (s_i + s_j)/2.0
            t_y = (s_i + s_j)/2.0
            
            dx = abs(cx_i - cx_j)
            dy = abs(cy_i - cy_j)
            
            # Overlap in x
            ox = t_x - dx
            # Overlap in y
            oy = t_y - dy
            
            # Smooth max
            soft_max_ox = (ox + np.logaddexp(0, -k*ox)) / k
            soft_max_oy = (oy + np.logaddexp(0, -k*oy)) / k
            
            # Penalty is product of overlaps (if both positive)
            # Using smooth approximation of max(0, ox) * max(0, oy)
            # Actually, if ox > 0 and oy > 0, they overlap.
            # We want to penalize this.
            # A simple smooth penalty: (soft_max_ox * soft_max_oy)
            # This is 0 if either is <= 0 (approximately).
            penalty += soft_max_ox * soft_max_oy
    
    return penalty

def _objective(x, n=10, lambda_pen=1000.0):
    """
    Objective: Minimize -sum(sides) + lambda * penalty
    """
    sides = x[2*n:]
    sum_sides = np.sum(sides)
    pen = _smooth_penalty(x, n)
    return -sum_sides + lambda_pen * pen

def _local_refine(centers, sides, tol=1e-7, deadline=None):
    """
    Local refinement using SLSQP.
    """
    n = len(sides)
    x0 = np.concatenate([centers.flatten(), sides])
    
    def obj(x):
        return _objective(x, n, lambda_pen=1000.0)
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0, 1)) # cx
        bounds.append((0, 1)) # cy
        bounds.append((1e-6, 1.0)) # s
    
    # We don't use constraints in SLSQP, we use penalty in objective.
    # This is a heuristic approach.
    
    result = minimize(
        obj, x0, method='SLSQP',
        bounds=bounds,
        options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
    )
    
    if result.success and np.all(np.isfinite(result.x)):
        new_centers = result.x[:2*n].reshape(n, 2)
        new_sides = result.x[2*n:]
        
        # Clip to valid range
        for i in range(n):
            new_sides[i] = max(1e-6, new_sides[i])
            new_centers[i, 0] = np.clip(new_centers[i, 0], new_sides[i]/2, 1 - new_sides[i]/2)
            new_centers[i, 1] = np.clip(new_centers[i, 1], new_sides[i]/2, 1 - new_sides[i]/2)
        
        if _is_feasible(new_centers, new_sides, tol):
            return new_centers, new_sides
    
    return centers, sides

def _global_search(start_time, deadline, n=10):
    """
    Run differential evolution to find a good starting point.
    """
    bounds = []
    for i in range(n):
        bounds.append((0, 1)) # cx
        bounds.append((0, 1)) # cy
        bounds.append((0.01, 0.5)) # s (limit max side to 0.5 to help convergence? Or 1.0)
        # Actually, sides can be up to 1.0.
        bounds[-1] = (0.01, 1.0)
    
    # DE is slow. Use a smaller population and maxiter.
    # We have ~500s budget.
    # DE with 30 vars, pop 30, maxiter 100.
    
    # Use a seed for reproducibility in testing, but random in production?
    # The task doesn't specify seed.
    
    def de_objective(x):
        return _objective(x, n, lambda_pen=1000.0)
    
    try:
        res = differential_evolution(
            de_objective, bounds,
            seed=42,
            maxiter=50,
            popsize=15,
            tol=1e-4,
            disp=False,
            polish=False # We will polish separately
        )
        
        if res.success or np.isfinite(res.fun):
            x_best = res.x
            centers = x_best[:2*n].reshape(n, 2)
            sides = x_best[2*n:]
            
            # Clip
            for i in range(n):
                sides[i] = max(1e-6, sides[i])
                centers[i, 0] = np.clip(centers[i, 0], sides[i]/2, 1 - sides[i]/2)
                centers[i, 1] = np.clip(centers[i, 1], sides[i]/2, 1 - sides[i]/2)
            
            if _is_feasible(centers, sides):
                return centers, sides
    except Exception:
        pass
    
    return None, None

def squares_in_square10():
    start_time = time.time()
    deadline = start_time + 580
    
    best_centers = None
    best_sides = None
    best_val = -np.inf
    
    # 1. Start with known good packing
    c1, s1 = _make_2big_8small()
    if _is_feasible(c1, s1):
        best_centers = c1.copy()
        best_sides = s1.copy()
        best_val = np.sum(s1)
    
    # 2. Local refinement of the known good packing
    if best_centers is not None and time.time() < deadline - 100:
        try:
            refined_centers, refined_sides = _local_refine(
                best_centers, best_sides, deadline=deadline
            )
            if _is_feasible(refined_centers, refined_sides):
                refined_val = np.sum(refined_sides)
                if refined_val > best_val:
                    best_val = refined_val
                    best_centers = refined_centers
                    best_sides = refined_sides
        except Exception:
            pass
    
    # 3. Global search to find potentially better topology
    if time.time() < deadline - 200:
        try:
            g_centers, g_sides = _global_search(start_time, deadline)
            if g_centers is not None and _is_feasible(g_centers, g_sides):
                g_val = np.sum(g_sides)
                if g_val > best_val:
                    best_val = g_val
                    best_centers = g_centers
                    best_sides = g_sides
        except Exception:
            pass
    
    # 4. Local refinement of the global search result (or best found)
    if best_centers is not None and time.time() < deadline - 50:
        try:
            # Run local refine again on the current best
            final_centers, final_sides = _local_refine(
                best_centers, best_sides, deadline=deadline
            )
            if _is_feasible(final_centers, final_sides):
                final_val = np.sum(final_sides)
                if final_val > best_val:
                    best_val = final_val
                    best_centers = final_centers
                    best_sides = final_sides
        except Exception:
            pass
    
    if best_centers is None:
        best_centers, best_sides = _make_2big_8small()
    
    return best_centers, best_sides
