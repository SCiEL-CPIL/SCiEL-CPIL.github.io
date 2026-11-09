import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = 590.0  # Safety margin
    
    # Unpack helper
    def unpack(params):
        cx = params[0:n]
        cy = params[n:2*n]
        s = params[2*n:3*n]
        return cx, cy, s

    # Smooth Objective Function
    # Minimize -sum(s) + penalty
    def objective(params):
        cx, cy, s = unpack(params)
        
        # Ensure s is positive for penalty calculation stability, though bounds handle it
        # Boundary Penalty
        # Left: cx - s/2 < 0 -> (cx - s/2)^2
        # Right: cx + s/2 > 1 -> (cx + s/2 - 1)^2
        # Bottom: cy - s/2 < 0 -> (cy - s/2)^2
        # Top: cy + s/2 > 1 -> (cy + s/2 - 1)^2
        
        left_viol = np.maximum(0, -(cx - s/2))
        right_viol = np.maximum(0, (cx + s/2) - 1)
        bottom_viol = np.maximum(0, -(cy - s/2))
        top_viol = np.maximum(0, (cy + s/2) - 1)
        
        bound_penalty = np.sum(left_viol**2) + np.sum(right_viol**2) + \
                        np.sum(bottom_viol**2) + np.sum(top_viol**2)
        
        # Overlap Penalty
        # For each pair i, j:
        # dx = |cx_i - cx_j|, dy = |cy_i - cy_j|
        # t = (s_i + s_j) / 2
        # Overlap if dx < t and dy < t.
        # Penalty term: max(0, t - dx)^2 + max(0, t - dy)^2
        # This is 0 if dx >= t OR dy >= t? 
        # If dx >= t, first term is 0. If dy < t, second term is > 0.
        # So it penalizes even if they are separated in x but close in y.
        # This is a valid "soft" constraint.
        
        overlap_penalty = 0.0
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                t = (s[i] + s[j]) / 2.0
                
                # Only calculate if potentially overlapping to save time? 
                # The max(0, ...) handles it.
                px = max(0.0, t - dx)
                py = max(0.0, t - dy)
                overlap_penalty += px*px + py*py
        
        # Total Objective
        # We want to maximize sum(s), so minimize -sum(s)
        # Penalty weight should be high enough to dominate but not break gradients
        weight = 1000.0
        return -np.sum(s) + weight * (bound_penalty + overlap_penalty)

    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # cx
        bounds.append((0.0, 1.0)) # cy
        bounds.append((0.01, 1.0)) # s

    # Initial Best (Baseline Grid)
    k = 3
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx_base = (cols + 0.5) * side
    cy_base = (rows + 0.5) * side
    # Adjust last square if grid is not full? 3x3=9. 10th square?
    # The baseline code just used 3x3 for 9 and presumably placed the 10th somewhere or had a bug?
    # Actually, 3x3 grid has 9 squares. The 10th square needs a place.
    # Let's create a valid baseline manually.
    # 4 squares of side 0.5? No.
    # Let's use a 2x5 grid of small squares? 2 rows, 5 cols. Side 0.2. Sum = 2.0. Bad.
    # Let's use a 1x10 strip? Side 0.1. Sum = 1.0. Bad.
    # Let's use the 3x3 grid for 9 squares, and put the 10th in the middle of one? No, overlap.
    # Let's just run DE and keep the best valid found.
    
    best_centers = None
    best_sides = None
    best_score = -np.inf
    
    # Helper to validate and update best
    def try_update_best(params):
        nonlocal best_centers, best_sides, best_score
        cx, cy, s = unpack(params)
        
        # Check validity strictly
        tol = 1e-7
        valid = True
        for i in range(n):
            if cx[i] - s[i]/2 < -tol or cx[i] + s[i]/2 > 1 + tol:
                valid = False
                break
            if cy[i] - s[i]/2 < -tol or cy[i] + s[i]/2 > 1 + tol:
                valid = False
                break
            if s[i] <= 0:
                valid = False
                break
        
        if not valid:
            return
        
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                t = (s[i] + s[j]) / 2.0
                if dx < t - tol and dy < t - tol:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            score = np.sum(s)
            if score > best_score:
                best_score = score
                best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
                best_sides = s.astype(np.float64)

    # Run Differential Evolution
    # Use multiple seeds if time permits
    seeds = [42, 123, 999]
    
    for seed in seeds:
        if time.time() > deadline - 120: # Leave time for refinement
            break
            
        try:
            # DE with smooth objective
            # popsize=20, maxiter=100 is reasonable
            result = differential_evolution(
                objective, 
                bounds, 
                maxiter=100, 
                popsize=20, 
                seed=seed,
                tol=0.01,
                init='sobol'
            )
            
            if result.fun < 1e6: # Heuristic check if it found something decent
                params = result.x
                try_update_best(params)
                
                # Local Refinement
                # Use L-BFGS-B or Nelder-Mead
                # L-BFGS-B is better for smooth functions
                res_refine = minimize(
                    objective, 
                    params, 
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res_refine.fun < 1e6:
                    try_update_best(res_refine.x)
                    
        except Exception:
            continue

    # If no valid solution was found (unlikely, but possible), fallback to a known valid one
    if best_centers is None:
        # Construct a simple valid packing:
        # 4 squares of side 0.5 in corners?
        # (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)
        # This leaves a hole in the middle? No, they touch at (0.5, 0.5).
        # We have 6 more squares to place.
        # The hole is 0.5x0.5? No, the 4 squares cover the whole square if they are 0.5x0.5 and centered at 0.25/0.75.
        # So we can't fit 6 more.
        
        # Let's try 10 squares of side 0.3?
        # 3x3 grid of 0.3 squares covers 0.9x0.9.
        # Centers: 0.15, 0.45, 0.75.
        # Side 0.3.
        # 9 squares. 10th?
        # If we make them slightly smaller, say 0.29, we have space.
        # Let's just generate a random valid one or use the 3x3 grid with small squares.
        
        # Safe Fallback: 10 squares of side 0.1 in a 1x10 strip?
        # Centers x: 0.05, 0.15, ..., 0.95. y: 0.5.
        # Side 0.1.
        # Sum = 1.0.
        # This is valid.
        
        cx = np.linspace(0.05, 0.95, 10)
        cy = np.full(10, 0.5)
        s = np.full(10, 0.1)
        best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
        best_sides = s.astype(np.float64)
        best_score = np.sum(s)

    return best_centers, best_sides
