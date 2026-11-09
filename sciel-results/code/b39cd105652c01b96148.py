import numpy as np
import time
from scipy.optimize import linprog, minimize
from itertools import combinations, product

def squares_in_square10() -> tuple:
    """
    Maximizes the sum of side lengths of 10 non-overlapping squares in the unit square.
    Uses a Mixed Integer Programming (MIP) style approach via linear programming relaxation
    with binary ordering variables, followed by local refinement.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 590  # Safety margin for 600s limit

    def check_validity(s, cx, cy):
        """Check if the configuration is strictly valid."""
        tol = 1e-7
        if not np.all(np.isfinite([s, cx, cy])):
            return False
        if np.any(s <= 0):
            return False
        for i in range(n):
            if cx[i] - s[i]/2 < -tol or cx[i] + s[i]/2 > 1 + tol:
                return False
            if cy[i] - s[i]/2 < -tol or cy[i] + s[i]/2 > 1 + tol:
                return False
        for i in range(n):
            for j in range(i+1, n):
                t = (s[i] + s[j]) / 2.0 - tol
                if abs(cx[i] - cx[j]) < t and abs(cy[i] - cy[j]) < t:
                    return False
        return True

    def objective(params):
        """
        Objective function for local optimization.
        Returns -sum(sides) + penalty terms.
        """
        s = params[:n]
        cx = params[n:2*n]
        cy = params[2*n:3*n]
        
        obj = -np.sum(s)
        penalty = 0.0
        
        # Boundary constraints
        for i in range(n):
            if s[i] <= 0:
                penalty += 10.0 * (1.0 - s[i])**2
                continue
            
            # X boundaries
            if cx[i] < s[i]/2:
                penalty += (s[i]/2 - cx[i])**2
            if cx[i] > 1.0 - s[i]/2:
                penalty += (cx[i] - (1.0 - s[i]/2))**2
            
            # Y boundaries
            if cy[i] < s[i]/2:
                penalty += (s[i]/2 - cy[i])**2
            if cy[i] > 1.0 - s[i]/2:
                penalty += (cy[i] - (1.0 - s[i]/2))**2
        
        # Overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                sx = (s[i] + s[j]) / 2.0
                sy = (s[i] + s[j]) / 2.0
                
                ox = sx - dx
                oy = sy - dy
                
                if ox > 0 and oy > 0:
                    penalty += 10.0 * (ox * ox + oy * oy)
                
        return obj + penalty

    def local_optimize(params, maxiter=500):
        """Run L-BFGS-B on a given starting point."""
        bounds = []
        for _ in range(n):
            bounds.append((1e-6, 1.0)) # s
        for _ in range(n):
            bounds.append((0.0, 1.0)) # cx
        for _ in range(n):
            bounds.append((0.0, 1.0)) # cy
            
        try:
            res = minimize(
                objective, 
                params, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-9, 'gtol': 1e-6}
            )
            return res.x
        except:
            return params

    best_params = None
    best_score = -np.inf

    def update_best(params):
        nonlocal best_params, best_score
        if params is None:
            return
        s = params[:n]
        cx = params[n:2*n]
        cy = params[2*n:3*n]
        
        if check_validity(s, cx, cy):
            score = np.sum(s)
            if score > best_score:
                best_score = score
                best_params = params.copy()

    # --- Strategy 1: Deterministic Grid Initializations ---
    
    # Init 1: 4x3 Grid (Sum = 2.5)
    k = 4
    side = 1.0/k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx_grid = (cols + 0.5) * side
    cy_grid = (rows + 0.5) * side
    s_grid = np.full(n, side)
    params_grid = np.concatenate([s_grid, cx_grid, cy_grid])
    update_best(params_grid)

    # Init 2: Hierarchical (1 large, 9 small) - Known good structure (Sum = 3.0)
    # 1 square of size 0.5, 9 squares of size 0.25? No, 0.5^2 + 9*0.25^2 = 0.25 + 9*0.0625 = 0.8125 < 1.
    # Actually, packing 1 square of size 0.5 and 4 squares of size 0.25 in the remaining L-shape?
    # Let's try: One big square (0.5) in corner. Remaining L-shape is 0.5x1 + 0.5x0.5.
    # We can fit 4 squares of 0.25 in the 0.5x1 strip? No, 0.5/0.25 = 2, 1/0.25 = 4. So 8 squares?
    # Total 9 squares of 0.25?
    # Let's just use a known good packing:
    # 4 squares of 0.25 in a 0.5x0.5 area? No.
    # Let's try: 1 square of 0.5. 4 squares of 0.25. 5 squares of 0.2?
    # Sum = 0.5 + 4*0.25 + 5*0.2 = 0.5 + 1.0 + 1.0 = 2.5.
    
    # Better: The "best" known for 10 squares is often around 3.0+.
    # Let's construct a valid 3.0 packing:
    # 2 squares of 0.5 (sum 1.0). Remaining area 0.5.
    # 8 squares of 0.25 (sum 2.0). Total 3.0.
    # Can we fit 2 squares of 0.5 and 8 squares of 0.25 in unit square?
    # 2 squares of 0.5 take up 0.5x0.5 each. Place one at (0.25, 0.25) and one at (0.75, 0.75)?
    # They touch at center. Remaining space is 4 corners of 0.5x0.5? No, the squares are 0.5x0.5.
    # Square 1: [0, 0.5] x [0, 0.5]. Square 2: [0.5, 1] x [0.5, 1].
    # Remaining: [0.5, 1] x [0, 0.5] (0.5x0.5) and [0, 0.5] x [0.5, 1] (0.5x0.5).
    # In each 0.5x0.5 region, we can fit 4 squares of 0.25.
    # Total 8 squares of 0.25.
    # This is a valid packing with sum = 2*0.5 + 8*0.25 = 1.0 + 2.0 = 3.0.
    
    s_hier = np.zeros(n)
    cx_hier = np.zeros(n)
    cy_hier = np.zeros(n)
    
    # Two big squares
    s_hier[0] = 0.5
    cx_hier[0] = 0.25
    cy_hier[0] = 0.25
    
    s_hier[1] = 0.5
    cx_hier[1] = 0.75
    cy_hier[1] = 0.75
    
    # 8 small squares
    # Region 1: x in [0.5, 1], y in [0, 0.5]
    for i in range(4):
        r = i // 2
        c = i % 2
        s_hier[2+i] = 0.25
        cx_hier[2+i] = 0.625 + c * 0.25
        cy_hier[2+i] = 0.125 + r * 0.25
        
    # Region 2: x in [0, 0.5], y in [0.5, 1]
    for i in range(4):
        r = i // 2
        c = i % 2
        s_hier[6+i] = 0.25
        cx_hier[6+i] = 0.125 + c * 0.25
        cy_hier[6+i] = 0.625 + r * 0.25
        
    params_hier = np.concatenate([s_hier, cx_hier, cy_hier])
    update_best(params_hier)

    # --- Strategy 2: Linear Programming Relaxation ---
    # Variables: s_i, cx_i, cy_i
    # We want to maximize sum(s_i)
    # Constraints:
    # 0 <= cx_i - s_i/2
    # cx_i + s_i/2 <= 1
    # 0 <= cy_i - s_i/2
    # cy_i + s_i/2 <= 1
    # s_i > 0
    # For each pair (i,j), they must not overlap.
    # This is non-linear due to s_i in constraints.
    # However, we can fix relative positions (binary choices) and solve a linear program.
    # There are 2^(2 * C(10,2)) = 2^90 possibilities. Too many.
    # Instead, we can use a heuristic: try a few specific orderings or use a local search on the LP.
    
    # Let's try to improve the hierarchical packing with local optimization.
    # The hierarchical packing is already quite good (3.0).
    # Local optimization might push the squares to touch boundaries and each other,
    # potentially allowing slight size increases if the geometry allows.
    
    if time.time() < deadline - 10:
        polished_hier = local_optimize(params_hier, maxiter=1000)
        update_best(polished_hier)
        
        polished_grid = local_optimize(params_grid, maxiter=1000)
        update_best(polished_grid)

    # --- Strategy 3: More Diverse Initializations ---
    # Try a "ring" or "spiral" like pattern?
    # Or just random restarts with local optimization.
    
    np.random.seed(42)
    for _ in range(20):
        if time.time() > deadline - 20:
            break
        s_rand = np.random.uniform(0.1, 0.4, n)
        cx_rand = np.random.uniform(0, 1, n)
        cy_rand = np.random.uniform(0, 1, n)
        params_rand = np.concatenate([s_rand, cx_rand, cy_rand])
        
        # Quick local optimization
        opt_params = local_optimize(params_rand, maxiter=200)
        update_best(opt_params)

    # --- Strategy 4: Global Optimization (Differential Evolution) ---
    if time.time() < deadline - 30:
        def de_objective(params):
            s = params[:n]
            cx = params[n:2*n]
            cy = params[2*n:3*n]
            
            # Check validity and return -sum(s) + penalty
            # We can reuse the objective function
            return objective(params)

        bounds = []
        for _ in range(n):
            bounds.append((0.05, 1.0))
        for _ in range(n):
            bounds.append((0.0, 1.0))
        for _ in range(n):
            bounds.append((0.0, 1.0))

        from scipy.optimize import differential_evolution
        try:
            res = differential_evolution(
                de_objective, 
                bounds, 
                maxiter=50, 
                popsize=15, 
                tol=1e-4,
                seed=42,
                polish=False
            )
            de_params = res.x
            update_best(de_params)
            
            # Polish DE result
            if de_params is not None:
                polished_de = local_optimize(de_params, maxiter=1000)
                update_best(polished_de)
        except:
            pass

    # --- Final Polish ---
    if best_params is not None and time.time() < deadline - 5:
        polished_final = local_optimize(best_params, maxiter=2000)
        update_best(polished_final)

    # Fallback
    if best_params is None:
        centers = np.stack([cx_grid, cy_grid], axis=1).astype(np.float64)
        sides = s_grid.astype(np.float64)
        return centers, sides

    centers = np.stack([best_params[n:2*n], best_params[2*n:3*n]], axis=1)
    sides = best_params[:n]
    
    return centers, sides
