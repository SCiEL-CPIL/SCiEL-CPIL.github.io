import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def squares_in_square10() -> tuple:
    """
    Maximizes the sum of side lengths of 10 non-overlapping squares in the unit square.
    Combines deterministic structural initialization, local gradient-based refinement,
    and global differential evolution search.
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
        Objective function for optimization.
        Returns -sum(sides) + penalty terms.
        """
        s = params[:n]
        cx = params[n:2*n]
        cy = params[2*n:3*n]
        
        obj = -np.sum(s)
        penalty = 0.0
        
        # Boundary constraints:
        # cx - s/2 >= 0  => cx >= s/2
        # cx + s/2 <= 1  => cx <= 1 - s/2
        # Same for cy
        
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
        
        # Overlap constraints:
        # For pair (i, j), if they overlap in both x and y, it's forbidden.
        # Overlap in x if |cx_i - cx_j| < (s_i + s_j)/2
        # Overlap in y if |cy_i - cy_j| < (s_i + s_j)/2
        # We want to push them apart if both conditions are met.
        
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                sx = (s[i] + s[j]) / 2.0
                sy = (s[i] + s[j]) / 2.0
                
                # Smooth approximation of max(0, overlap)
                # Using a soft max or just squared overlap if positive
                ox = sx - dx
                oy = sy - dy
                
                if ox > 0 and oy > 0:
                    # Penalty proportional to squared overlap area
                    penalty += 10.0 * (ox * ox + oy * oy)
                
        return obj + penalty

    def local_optimize(params):
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
                options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-6}
            )
            return res.x
        except:
            return params

    def global_optimize():
        """Run Differential Evolution."""
        bounds = []
        for _ in range(n):
            bounds.append((0.05, 1.0)) # s
        for _ in range(n):
            bounds.append((0.0, 1.0)) # cx
        for _ in range(n):
            bounds.append((0.0, 1.0)) # cy
            
        try:
            res = differential_evolution(
                objective, 
                bounds, 
                maxiter=100, 
                popsize=10, 
                tol=1e-6,
                seed=42
            )
            return res.x
        except:
            return None

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

    # 1. Generate diverse initializations
    
    # Init 1: Grid
    k = 4
    side = 1.0/k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx_grid = (cols + 0.5) * side
    cy_grid = (rows + 0.5) * side
    s_grid = np.full(n, side)
    params_grid = np.concatenate([s_grid, cx_grid, cy_grid])
    
    # Init 2: Hierarchical (1 large, 9 small)
    s_hier = np.zeros(n)
    cx_hier = np.zeros(n)
    cy_hier = np.zeros(n)
    s_hier[0] = 0.5
    cx_hier[0] = 0.25
    cy_hier[0] = 0.25
    
    # Place 9 small squares in the L-shape
    # Top strip: y in [0.5, 1], x in [0, 1] -> 4 squares of 0.25
    for i in range(4):
        s_hier[1+i] = 0.25
        cx_hier[1+i] = 0.125 + i * 0.25
        cy_hier[1+i] = 0.75
        
    # Bottom-Right 0.5x0.5 square: 4 squares of 0.25
    for i in range(4):
        row = i // 2
        col = i % 2
        s_hier[5+i] = 0.25
        cx_hier[5+i] = 0.625 + col * 0.25
        cy_hier[5+i] = 0.125 + row * 0.25
        
    # 10th square (index 9) - place in center, will be moved by optimizer
    s_hier[9] = 0.25
    cx_hier[9] = 0.5
    cy_hier[9] = 0.5
    
    params_hier = np.concatenate([s_hier, cx_hier, cy_hier])
    
    # Init 3: Random
    np.random.seed(42)
    params_rand = np.concatenate([
        np.random.uniform(0.1, 0.4, n),
        np.random.uniform(0, 1, n),
        np.random.uniform(0, 1, n)
    ])

    inits = [params_grid, params_hier, params_rand]
    
    # 2. Local optimization from initializations
    for init in inits:
        if time.time() > deadline - 10:
            break
        opt_params = local_optimize(init)
        update_best(opt_params)
        
        # Polish if valid
        if best_params is not None and best_score > 0:
            # Try to polish the best found so far
            polished = local_optimize(best_params)
            update_best(polished)

    # 3. Global optimization (Differential Evolution)
    if time.time() < deadline - 30:
        de_params = global_optimize()
        update_best(de_params)
        
        # Polish DE result
        if de_params is not None:
            polished = local_optimize(de_params)
            update_best(polished)

    # 4. Final polish of the absolute best
    if best_params is not None and time.time() < deadline - 10:
        # Run a tighter local optimization
        bounds = []
        for _ in range(n):
            bounds.append((1e-6, 1.0))
        for _ in range(n):
            bounds.append((0.0, 1.0))
        for _ in range(n):
            bounds.append((0.0, 1.0))
            
        try:
            res_final = minimize(
                objective, 
                best_params, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            update_best(res_final.x)
        except:
            pass

    # Fallback
    if best_params is None:
        centers = np.stack([cx_grid, cy_grid], axis=1).astype(np.float64)
        sides = s_grid.astype(np.float64)
        return centers, sides

    centers = np.stack([best_params[n:2*n], best_params[2*n:3*n]], axis=1)
    sides = best_params[:n]
    
    return centers, sides
