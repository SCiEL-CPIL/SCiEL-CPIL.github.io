import numpy as np
from scipy.optimize import minimize
import time

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 590.0  # 10s safety margin
    
    def penalty_and_objective(x):
        # x: [s_0..s_9, cx_0..cx_9, cy_0..cy_9]
        s = x[:n]
        cx = x[n:2*n]
        cy = x[2*n:3*n]
        
        # Sides must be positive
        s_pos = np.maximum(s, 0.0)
        
        # Boundary violations
        b_pen = 0.0
        for i in range(n):
            b_pen += max(0.0, s_pos[i]/2.0 - cx[i])
            b_pen += max(0.0, s_pos[i]/2.0 - (1.0 - cx[i]))
            b_pen += max(0.0, s_pos[i]/2.0 - cy[i])
            b_pen += max(0.0, s_pos[i]/2.0 - (1.0 - cy[i]))
            
        # Overlap violations
        o_pen = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = (s_pos[i] + s_pos[j]) / 2.0
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                if dx < d and dy < d:
                    # Smooth approximation or just product
                    # Product is 0 if separated, >0 if overlapping
                    o_pen += (d - dx) * (d - dy)
                    
        # Objective: Maximize sum(s) -> Minimize -sum(s) + Lambda * Penalty
        # Lambda should be large enough to force feasibility
        # Sum(s) is at most ~4. Penalty terms are O(1).
        # If Lambda=100, penalty dominates.
        obj = -np.sum(s_pos) + 100.0 * (b_pen + o_pen)
        return obj

    def is_valid(centers, sides, tol=1e-7):
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if np.any(sides <= 0):
            return False
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s/2 < -tol or cx + s/2 > 1 + tol:
                return False
            if cy - s/2 < -tol or cy + s/2 > 1 + tol:
                return False
        for i in range(n):
            for j in range(i+1, n):
                cx_i, cy_i = centers[i]
                cx_j, cy_j = centers[j]
                s_i = sides[i]
                s_j = sides[j]
                t = (s_i + s_j) / 2.0 - tol
                if abs(cx_i - cx_j) < t and abs(cy_i - cy_j) < t:
                    return False
        return True

    best_score = -np.inf
    best_centers = None
    best_sides = None
    
    # Initial configurations
    inits = []
    
    # 1. Grid 3x3 + 1
    k = 3
    side = 1.0/k
    idx = np.arange(9)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    # Add 10th square, small, in center?
    cx10 = 0.5
    cy10 = 0.5
    s10 = 0.1
    cx_full = np.concatenate([cx, [cx10]])
    cy_full = np.concatenate([cy, [cy10]])
    s_full = np.full(n, side)
    s_full[9] = s10
    x0 = np.concatenate([s_full, cx_full, cy_full])
    inits.append(x0)
    
    # 2. Random
    np.random.seed(42)
    for _ in range(20):
        s_rand = np.random.uniform(0.1, 0.5, n)
        cx_rand = np.random.uniform(0.2, 0.8, n)
        cy_rand = np.random.uniform(0.2, 0.8, n)
        x_rand = np.concatenate([s_rand, cx_rand, cy_rand])
        inits.append(x_rand)

    # 3. Known good heuristic: One big, rest small
    # Big square 0.5 in corner
    s_heur = np.full(n, 0.2)
    s_heur[0] = 0.5
    cx_heur = np.full(n, 0.5)
    cy_heur = np.full(n, 0.5)
    cx_heur[0] = 0.25
    cy_heur[0] = 0.25
    # Others pushed away
    cx_heur[1] = 0.75; cy_heur[1] = 0.25
    cx_heur[2] = 0.25; cy_heur[2] = 0.75
    cx_heur[3] = 0.75; cy_heur[3] = 0.75
    x_heur = np.concatenate([s_heur, cx_heur, cy_heur])
    inits.append(x_heur)

    for x0 in inits:
        if time.time() > deadline:
            break
            
        # Run Nelder-Mead
        # Set maxiter and maxfev to limit time per run
        # ~1000 iterations should be fast (<1s)
        res = minimize(penalty_and_objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'maxfev': 4000, 'xatol': 1e-8, 'fatol': 1e-8})
        
        x_opt = res.x
        s_opt = np.maximum(x_opt[:n], 1e-6) # Ensure positive
        cx_opt = x_opt[n:2*n]
        cy_opt = x_opt[2*n:3*n]
        
        centers = np.stack([cx_opt, cy_opt], axis=1)
        sides = s_opt
        
        if is_valid(centers, sides):
            score = np.sum(sides)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                best_sides = sides.copy()

    if best_centers is None:
        # Fallback to grid
        k = 3
        side = 1.0/k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)
        return centers, sides
        
    return best_centers, best_sides
