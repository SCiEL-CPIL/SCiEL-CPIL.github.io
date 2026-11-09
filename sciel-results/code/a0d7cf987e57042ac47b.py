import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = 600.0
    
    # Baseline
    k = int(np.ceil(np.sqrt(n)))
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx_base = (cols + 0.5) * side
    cy_base = (rows + 0.5) * side
    centers_base = np.stack([cx_base, cy_base], axis=1).astype(np.float64)
    sides_base = np.full(n, side, dtype=np.float64)
    best_score = np.sum(sides_base)
    best_centers = centers_base
    best_sides = sides_base

    # Optimization
    # Variables: cx1, cy1, s1, cx2, cy2, s2, ...
    # Total 3 * n variables
    
    def unpack(params):
        cx = params[0:n]
        cy = params[n:2*n]
        s = params[2*n:3*n]
        return cx, cy, s

    def objective(params):
        cx, cy, s = unpack(params)
        
        # Boundary check
        # If any square goes out of bounds, return penalty
        for i in range(n):
            if cx[i] - s[i]/2 < -1e-7 or cx[i] + s[i]/2 > 1 + 1e-7:
                return 1e10 # Penalty
            if cy[i] - s[i]/2 < -1e-7 or cy[i] + s[i]/2 > 1 + 1e-7:
                return 1e10
        
        # Overlap check
        for i in range(n):
            for j in range(i+1, n):
                # Overlap if |dx| < (si+sj)/2 and |dy| < (si+sj)/2
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                t = (s[i] + s[j]) / 2.0
                if dx < t - 1e-7 and dy < t - 1e-7:
                    return 1e10
        
        # If valid, return negative sum of sides (to minimize)
        return -np.sum(s)

    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # cx
        bounds.append((0.0, 1.0)) # cy
        bounds.append((0.01, 1.0)) # s

    # Run Differential Evolution
    # maxiter and popsize control runtime
    # 30 dimensions. popsize=20, maxiter=100 is ~ 6000 evaluations.
    # Each eval is O(n^2) = 100 ops. Very fast.
    
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=200, 
            popsize=30, 
            seed=42,
            tol=0.01,
            init='sobol'
        )
        
        if result.fun < 1e9: # Valid solution found
            params = result.x
            cx, cy, s = unpack(params)
            score = np.sum(s)
            if score > best_score:
                best_centers = np.stack([cx, cy], axis=1)
                best_sides = s
                best_score = score
                
    except Exception as e:
        pass

    # Local Refinement with Nelder-Mead on the best found
    if best_score > np.sum(sides_base):
        try:
            res_nm = minimize(
                objective, 
                np.hstack([best_centers[:,0], best_centers[:,1], best_sides]),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res_nm.fun < 1e9:
                params = res_nm.x
                cx, cy, s = unpack(params)
                score = np.sum(s)
                if score > best_score:
                    best_centers = np.stack([cx, cy], axis=1)
                    best_sides = s
                    best_score = score
        except Exception:
            pass

    # Final validation and return
    # Ensure finite and positive
    if not np.all(np.isfinite(best_centers)) or not np.all(np.isfinite(best_sides)):
        return centers_base, sides_base
    if not np.all(best_sides > 0):
        return centers_base, sides_base

    return best_centers, best_sides
