import numpy as np
import time
import random
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    """
    Maximizes the sum of side lengths of 10 non-overlapping squares in the unit square.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 600  # 600s limit, keep safety margin
    
    # Penalty function
    def penalty_and_objective(params):
        # params: 3n vector: s_0..s_9, cx_0..cx_9, cy_0..cy_9
        s = params[:n]
        cx = params[n:2*n]
        cy = params[2*n:3*n]
        
        # Objective: maximize sum(s) -> minimize -sum(s) + penalty
        obj = -np.sum(s)
        
        # Boundary penalty
        # Squares must be in [0, 1]
        # min_x = cx - s/2 >= 0 -> cx >= s/2
        # max_x = cx + s/2 <= 1 -> cx <= 1 - s/2
        # Similarly for y
        bound_pen = 0.0
        for i in range(n):
            if s[i] <= 0:
                bound_pen += 10.0 * (1.0 - s[i])**2 # Push s to be positive
            else:
                # x boundaries
                if cx[i] < s[i]/2:
                    bound_pen += (s[i]/2 - cx[i])**2
                if cx[i] > 1.0 - s[i]/2:
                    bound_pen += (cx[i] - (1.0 - s[i]/2))**2
                # y boundaries
                if cy[i] < s[i]/2:
                    bound_pen += (s[i]/2 - cy[i])**2
                if cy[i] > 1.0 - s[i]/2:
                    bound_pen += (cy[i] - (1.0 - s[i]/2))**2
        
        # Overlap penalty
        # For each pair i, j
        overlap_pen = 0.0
        for i in range(n):
            for j in range(i+1, n):
                # Check x overlap
                dx = abs(cx[i] - cx[j])
                sx = (s[i] + s[j]) / 2.0
                if dx < sx:
                    # Overlap in x
                    ox = sx - dx
                    # Check y overlap
                    dy = abs(cy[i] - cy[j])
                    sy = (s[i] + s[j]) / 2.0
                    if dy < sy:
                        oy = sy - dy
                        # Both overlap
                        # Penalty proportional to area of intersection or just sum of overlaps
                        # Using squared overlap for smoothness
                        overlap_pen += (ox * ox) + (oy * oy)
        
        total_pen = bound_pen + overlap_pen
        return obj + 10.0 * total_pen # Weight for penalty

    def gradient(params):
        # Numerical gradient is slow, but analytical is complex.
        # For robustness, we might use BFGS which estimates gradients, or Nelder-Mead.
        # Given the small dimension (30), Nelder-Mead might be okay but slow.
        # L-BFGS-B with numerical gradient is a good middle ground.
        # However, implementing analytical gradient is better.
        # Let's stick to numerical gradient with L-BFGS-B for simplicity and reliability.
        # Or use COBYLA which doesn't need gradients but is slower.
        # Let's try L-BFGS-B with numerical gradient.
        return None # Let scipy handle it if we don't provide it

    # Initializations
    inits = []
    
    # 1. Grid initialization (Baseline-like)
    k = 4
    side = 1.0/k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    s = np.full(n, side)
    params_grid = np.concatenate([s, cx, cy])
    inits.append(params_grid)
    
    # 2. Hierarchical 1-large, 9-small
    s_hier = np.zeros(n)
    cx_hier = np.zeros(n)
    cy_hier = np.zeros(n)
    s_hier[0] = 0.5
    cx_hier[0] = 0.25
    cy_hier[0] = 0.25
    
    # Place 9 small squares in the remaining L-shape
    # Top strip: y in [0.5, 1], x in [0, 1]
    # Right strip: x in [0.5, 1], y in [0, 0.5]
    # We'll place them roughly and let optimizer fix.
    small_side = 0.25
    s_hier[1:] = small_side
    
    # Top 4
    for i in range(4):
        s_hier[1+i] = 0.25
        cx_hier[1+i] = 0.125 + i * 0.25
        cy_hier[1+i] = 0.75
        
    # Bottom-Right 4 (in the 0.5x0.5 square at bottom right)
    for i in range(4):
        row = i // 2
        col = i % 2
        s_hier[5+i] = 0.25
        cx_hier[5+i] = 0.625 + col * 0.25
        cy_hier[5+i] = 0.125 + row * 0.25
        
    # 9th square? We have 1 + 4 + 4 = 9. We need 10.
    # The 9th small square (index 9) is currently uninitialized in the loop above?
    # Loop 1: 1..4 (4 squares)
    # Loop 2: 5..8 (4 squares)
    # Index 9 is missing.
    # Let's place index 9 in the middle of the L-shape or overlap and let optimizer fix.
    s_hier[9] = 0.25
    cx_hier[9] = 0.5
    cy_hier[9] = 0.5 # This will overlap with others, optimizer must fix
    
    params_hier = np.concatenate([s_hier, cx_hier, cy_hier])
    inits.append(params_hier)
    
    # 3. Random initializations
    for _ in range(20):
        s_rand = np.random.uniform(0.1, 0.4, n)
        cx_rand = np.random.uniform(0, 1, n)
        cy_rand = np.random.uniform(0, 1, n)
        # Clip to ensure some feasibility initially to help optimizer
        # Not strictly necessary if penalty is strong
        params_rand = np.concatenate([s_rand, cx_rand, cy_rand])
        inits.append(params_rand)

    best_params = None
    best_score = -np.inf

    for init in inits:
        if time.time() > deadline - 10:
            break
            
        # Bounds:
        # s > 0
        # cx in [0, 1]
        # cy in [0, 1]
        bounds = []
        for i in range(n):
            bounds.append((1e-6, 1.0)) # s
        for i in range(n):
            bounds.append((0.0, 1.0)) # cx
        for i in range(n):
            bounds.append((0.0, 1.0)) # cy

        # Optimize
        try:
            res = minimize(
                penalty_and_objective, 
                init, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-6}
            )
            
            if res.success:
                params = res.x
                # Calculate raw score
                s = params[:n]
                raw_score = np.sum(s)
                
                # Verify validity strictly
                valid = True
                cx = params[n:2*n]
                cy = params[2*n:3*n]
                tol = 1e-7
                for i in range(n):
                    if s[i] <= 0:
                        valid = False
                        break
                    if cx[i] - s[i]/2 < -tol or cx[i] + s[i]/2 > 1 + tol:
                        valid = False
                        break
                    if cy[i] - s[i]/2 < -tol or cy[i] + s[i]/2 > 1 + tol:
                        valid = False
                        break
                if valid:
                    for i in range(n):
                        for j in range(i+1, n):
                            t = (s[i] + s[j]) / 2.0 - tol
                            if abs(cx[i] - cx[j]) < t and abs(cy[i] - cy[j]) < t:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid and raw_score > best_score:
                    best_score = raw_score
                    best_params = params.copy()
                    
        except Exception as e:
            pass

    # Polish the best solution found
    if best_params is not None:
        # Run a few more iterations with tighter tolerance on the best solution
        try:
            res_polish = minimize(
                penalty_and_objective, 
                best_params, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res_polish.success:
                params = res_polish.x
                s = params[:n]
                raw_score = np.sum(s)
                cx = params[n:2*n]
                cy = params[2*n:3*n]
                
                # Re-verify
                valid = True
                tol = 1e-7
                for i in range(n):
                    if s[i] <= 0:
                        valid = False
                        break
                    if cx[i] - s[i]/2 < -tol or cx[i] + s[i]/2 > 1 + tol:
                        valid = False
                        break
                    if cy[i] - s[i]/2 < -tol or cy[i] + s[i]/2 > 1 + tol:
                        valid = False
                        break
                if valid:
                    for i in range(n):
                        for j in range(i+1, n):
                            t = (s[i] + s[j]) / 2.0 - tol
                            if abs(cx[i] - cx[j]) < t and abs(cy[i] - cy[j]) < t:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid and raw_score > best_score:
                    best_score = raw_score
                    best_params = params.copy()
        except Exception:
            pass

    if best_params is None:
        # Fallback to grid
        k = 4
        side = 1.0/k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)
        return centers, sides

    centers = np.stack([best_params[n:2*n], best_params[2*n:3*n]], axis=1)
    sides = best_params[:n]
    
    return centers, sides
