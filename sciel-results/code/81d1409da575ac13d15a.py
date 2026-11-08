import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 11
_H = np.sqrt(3) / 2.0

# Precompute all C(11,3)=165 triangle index triples
_combos = list(combinations(range(N), 3))
_ci = np.array([c[0] for c in _combos])
_cj = np.array([c[1] for c in _combos])
_ck = np.array([c[2] for c in _combos])

def _min_area(pts):
    """Minimum area of any triangle formed by 3 of the 11 points."""
    pi = pts[_ci]
    pj = pts[_cj]
    pk = pts[_ck]
    v1 = pj - pi
    v2 = pk - pi
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return float(np.min(areas))

def _inside_triangle(pts, tol=1e-9):
    """Check all points are inside (or on boundary of) the equilateral triangle."""
    x = pts[:, 0]
    y = pts[:, 1]
    if np.any(y < -tol):
        return False
    if np.any(y > _H * x + tol):
        return False
    if np.any(y > _H * (1.0 - x) + tol):
        return False
    if np.any(x < -tol) or np.any(x > 1.0 + tol):
        return False
    return True

def _project(pts):
    """Project points onto the triangle (clamp to feasible region)."""
    pts = pts.copy()
    x = pts[:, 0]
    y = pts[:, 1]
    x = np.clip(x, 0.0, 1.0)
    y = np.maximum(y, 0.0)
    upper = np.minimum(_H * x, _H * (1.0 - x))
    y = np.minimum(y, upper)
    pts[:, 0] = x
    pts[:, 1] = y
    return pts

def _soft_min_objective(flat_pts, k):
    """
    Smooth approximation of -min(area).
    Uses log-sum-exp trick for stability.
    """
    pts = flat_pts.reshape(N, 2)
    # Ensure points stay roughly in bounds during optimization steps
    # L-BFGS-B handles bounds, but projection helps if bounds are violated slightly in internal steps
    # However, standard L-BFGS-B with bounds is preferred. We will use bounds in minimize.
    
    pi = pts[_ci]
    pj = pts[_cj]
    pk = pts[_ck]
    v1 = pj - pi
    v2 = pk - pi
    # 2 * Area = cross product magnitude
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min: -1/k * log(sum(exp(-k * areas)))
    # To avoid overflow/underflow, subtract max area (which corresponds to min -k*area)
    # max_exp_term = -k * min(areas)
    # log_sum_exp = max_exp_term + log(sum(exp(-k * (areas - min_area))))
    # Objective = -1/k * (max_exp_term + log_sum) = min_area - 1/k * log(sum(exp(-k * (areas - min_area))))
    # As k -> inf, this approaches min_area.
    # We want to MINIMIZE the negative of this, so maximize min_area.
    # Objective to minimize: -soft_min_area
    
    # Numerical stability:
    # Let m = min(areas).
    # areas_shifted = areas - m
    # exp_terms = exp(-k * areas_shifted)
    # soft_min_val = m - (1/k) * log(sum(exp_terms))
    # We return -soft_min_val
    
    m = np.min(areas)
    if m == 0:
        # Degenerate case, penalize heavily
        return 1e6
    
    areas_shifted = areas - m
    # If k is large, -k * areas_shifted can be very negative -> exp is 0.
    # Only the min area (shifted to 0) contributes exp(0)=1.
    # Sum is approx 1 + sum(other small terms). Log is approx 0.
    # soft_min_val approx m.
    
    # Clip to prevent exp overflow if k is small and areas are large? 
    # areas are bounded by triangle area ~ 0.433. k might be up to 1000.
    # -k * (areas - m) <= 0. So exp is <= 1. No overflow.
    
    exp_terms = np.exp(-k * areas_shifted)
    log_sum = np.log(np.sum(exp_terms))
    soft_min_val = m - (1.0 / k) * log_sum
    
    return -soft_min_val

def _initial_config_symmetric():
    """
    Symmetric arrangement based on literature/heuristics for 11 points.
    """
    h = _H
    pts = np.array([
        [0.0, 0.0],       # 0: BL
        [1.0, 0.0],       # 1: BR
        [0.5, h],         # 2: Top
        [0.5, 0.0],       # 3: Bottom Mid
        [0.75, h/2.0],    # 4: Right Mid
        [0.25, h/2.0],    # 5: Left Mid
        [0.5, h/3.0],     # 6: Center
        [1/6.0, h/6.0],   # 7: BL Centroid
        [5/6.0, h/6.0],   # 8: BR Centroid
        [0.5, 2*h/3.0],   # 9: Top Centroid
        [0.625, h/4.0]    # 10: Extra
    ])
    return _project(pts)

def _local_optimize_smooth(pts, k_start, k_end, max_time, rng):
    """
    Optimize using L-BFGS-B with increasing k (annealing).
    """
    t0 = time.time()
    current_pts = pts.copy()
    
    # Bounds for L-BFGS-B
    # x: [0, 1]
    # y: [0, _H] (roughly, tighter bounds depend on x, but L-BFGS-B needs fixed bounds)
    # We can use loose bounds and rely on projection/penalty, or tight bounds.
    # Tight bounds for y are tricky because they depend on x.
    # Standard trick: optimize in barycentric coordinates? 
    # Or just use loose bounds [0,1]x[0,1] and penalize outside?
    # L-BFGS-B with bounds [0,1]x[0,1] is safe. The objective will naturally push points inside 
    # if we add a penalty for being outside, OR we just accept that it might wander slightly 
    # and project at the end. 
    # Better: Use bounds [0, 1] for x and [0, _H] for y.
    # But y > _H*x is invalid. 
    # Let's stick to [0,1]x[0,1] bounds and rely on the fact that the optimum is inside.
    # To ensure validity, we will project the result.
    
    bounds = [(0, 1), (0, _H)] * N
    
    # Annealing schedule for k
    # Start with small k (smooth, easy to navigate) to large k (sharp, precise min)
    num_steps = 5
    k_values = np.geomspace(k_start, k_end, num_steps)
    
    for k in k_values:
        if time.time() - t0 > max_time:
            break
            
        # Define objective for this k
        def obj(flat_pts):
            return _soft_min_objective(flat_pts, k)
        
        try:
            res = minimize(
                obj,
                current_pts.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res.success or res.fun < 1e5: # Check for sanity
                new_pts = res.x.reshape(N, 2)
                new_pts = _project(new_pts)
                # Only accept if it improves the true min area
                if _min_area(new_pts) >= _min_area(current_pts) - 1e-9:
                    current_pts = new_pts
        except Exception:
            pass
            
    return current_pts, _min_area(current_pts)

def construct():
    t_start = time.time()
    deadline = t_start + 140.0 # 10s buffer
    
    rng = np.random.RandomState(42)
    
    best_pts = _initial_config_symmetric()
    best_score = _min_area(best_pts)
    
    # Multi-start with smooth optimization
    # Strategy:
    # 1. One high-quality run from symmetric start with long budget.
    # 2. Several random restarts with shorter budgets.
    
    attempts = 0
    max_attempts = 10
    
    # List of k_start, k_end, budget_fraction
    schedules = [
        (10.0, 500.0, 0.4),   # Main run: significant time
        (10.0, 500.0, 0.1),   # Restart 1
        (10.0, 500.0, 0.1),   # Restart 2
        (10.0, 500.0, 0.1),   # Restart 3
        (10.0, 500.0, 0.1),   # Restart 4
        (10.0, 500.0, 0.1),   # Restart 5
        (10.0, 500.0, 0.1),   # Restart 6
        (10.0, 500.0, 0.1),   # Restart 7
        (10.0, 500.0, 0.1),   # Restart 8
        (10.0, 500.0, 0.1),   # Restart 9
    ]
    
    total_budget = 130.0
    
    for k_s, k_e, frac in schedules:
        time_left = deadline - time.time()
        if time_left < 5.0:
            break
            
        budget = min(total_budget * frac, time_left - 2.0)
        if budget < 2.0:
            break
            
        # Generate start point
        if attempts == 0:
            start_pts = best_pts.copy()
        else:
            # Perturb best or random
            if attempts % 2 == 0:
                start_pts = best_pts + rng.randn(N, 2) * 0.1
            else:
                # Random points inside triangle
                start_pts = np.random.rand(N, 2)
                # Map to triangle
                # Barycentric coords
                u = np.random.rand(N)
                v = np.random.rand(N)
                mask = u + v > 1.0
                u[mask] = 1.0 - u[mask]
                v[mask] = 1.0 - v[mask]
                start_pts[:, 0] = u
                start_pts[:, 1] = v * _H
                # This maps to the triangle (0,0), (1,0), (0.5, H) ?
                # Point = u*(1,0) + v*(0.5, H) ? No.
                # Standard mapping: P = A + u*(B-A) + v*(C-A)
                # A=(0,0), B=(1,0), C=(0.5, H)
                # P = u*(1,0) + v*(0.5, H) = (u + 0.5v, vH)
                # This requires u,v >= 0 and u+v <= 1.
                start_pts[:, 0] = u + 0.5 * v
                start_pts[:, 1] = v * _H
            
            start_pts = _project(start_pts)
        
        opt_pts, opt_score = _local_optimize_smooth(start_pts, k_s, k_e, budget, rng)
        
        if opt_score > best_score:
            best_score = opt_score
            best_pts = opt_pts.copy()
            
        attempts += 1
        
    # Final validation
    best_pts = _project(best_pts)
    if not _inside_triangle(best_pts):
        best_pts = _project(_initial_config_symmetric())
        
    return best_pts
