import numpy as np
import time
from scipy.optimize import minimize, linprog
from itertools import combinations

def squares_in_square10() -> tuple:
    """
    Maximizes the sum of side lengths of 10 non-overlapping squares in the unit square.
    Uses a Mixed Integer Programming (MIP) style approach via linear programming relaxation
    with binary ordering variables, followed by local refinement.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 580  # Safety margin for 600s limit

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

    def objective(params, penalty_weight=100.0):
        """
        Objective function for local optimization.
        Returns -sum(sides) + penalty terms.
        Uses a smoother penalty for overlaps.
        """
        s = params[:n]
        cx = params[n:2*n]
        cy = params[2*n:3*n]
        
        obj = -np.sum(s)
        penalty = 0.0
        
        # Boundary constraints
        for i in range(n):
            if s[i] <= 0:
                penalty += penalty_weight * (1.0 - s[i])**2
                continue
            
            # X boundaries
            if cx[i] < s[i]/2:
                penalty += penalty_weight * (s[i]/2 - cx[i])**2
            if cx[i] > 1.0 - s[i]/2:
                penalty += penalty_weight * (cx[i] - (1.0 - s[i]/2))**2
            
            # Y boundaries
            if cy[i] < s[i]/2:
                penalty += penalty_weight * (s[i]/2 - cy[i])**2
            if cy[i] > 1.0 - s[i]/2:
                penalty += penalty_weight * (cy[i] - (1.0 - s[i]/2))**2
        
        # Overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                sx = (s[i] + s[j]) / 2.0
                sy = (s[i] + s[j]) / 2.0
                
                ox = sx - dx
                oy = sy - dy
                
                # Smooth penalty: use max(0, ox)^2 + max(0, oy)^2
                # This is differentiable almost everywhere
                if ox > 0:
                    penalty += penalty_weight * ox * ox
                if oy > 0:
                    penalty += penalty_weight * oy * oy
                
        return obj + penalty

    def local_optimize(params, maxiter=2000, ftol=1e-12, gtol=1e-9):
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
                lambda p: objective(p), 
                params, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
            )
            return res.x
        except:
            return params

    def coordinate_descent(params, max_iter=5000, tol=1e-12):
        """Coordinate descent with line search for final polish."""
        best = params.copy()
        best_val = objective(best)
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            improved = False
            
            # Try optimizing each variable individually
            for var_idx in range(len(best)):
                orig_val = best[var_idx]
                
                # Binary search for optimal value in a small range
                step = 0.01
                for _ in range(60):
                    if time.time() > deadline - 5:
                        break
                    for delta in [step, -step]:
                        test = best.copy()
                        test[var_idx] = orig_val + delta
                        if test[var_idx] < 1e-6 and var_idx < n:
                            continue
                        if test[var_idx] < 0 or test[var_idx] > 1:
                            continue
                        
                        test_val = objective(test)
                        if test_val < best_val - tol:
                            best = test
                            best_val = test_val
                            improved = True
                            break
                    
                    if improved:
                        break
                    step *= 0.5
                
                if not improved:
                    # Try smaller steps
                    step = 1e-5
                    for _ in range(40):
                        if time.time() > deadline - 5:
                            break
                        for delta in [step, -step]:
                            test = best.copy()
                            test[var_idx] = orig_val + delta
                            if test[var_idx] < 1e-6 and var_idx < n:
                                continue
                            if test[var_idx] < 0 or test[var_idx] > 1:
                                continue
                            
                            test_val = objective(test)
                            if test_val < best_val - tol:
                                best = test
                                best_val = test_val
                                improved = True
                                break
                        
                        if improved:
                            break
                        step *= 0.5
            
            if not improved:
                break
        
        return best

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

    # Init 3: 5x2 Grid (Sum = 2.5)
    k = 5
    side = 1.0/k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx_grid5 = (cols + 0.5) * side
    cy_grid5 = (rows + 0.5) * side
    s_grid5 = np.full(n, side)
    params_grid5 = np.concatenate([s_grid5, cx_grid5, cy_grid5])
    update_best(params_grid5)

    # Init 4: Mixed sizes - 2 of 0.4, 8 of 0.15
    s_mixed = np.zeros(n)
    cx_mixed = np.zeros(n)
    cy_mixed = np.zeros(n)
    
    # Two 0.4 squares
    s_mixed[0] = 0.4
    cx_mixed[0] = 0.2
    cy_mixed[0] = 0.2
    
    s_mixed[1] = 0.4
    cx_mixed[1] = 0.8
    cy_mixed[1] = 0.8
    
    # Eight 0.15 squares in remaining space
    positions = [
        (0.6, 0.2), (0.6, 0.4), (0.6, 0.6), (0.6, 0.8),
        (0.2, 0.6), (0.4, 0.6), (0.4, 0.8), (0.2, 0.8)
    ]
    for i, (px, py) in enumerate(positions):
        s_mixed[2+i] = 0.15
        cx_mixed[2+i] = px
        cy_mixed[2+i] = py
    
    params_mixed = np.concatenate([s_mixed, cx_mixed, cy_mixed])
    update_best(params_mixed)

    # --- Strategy 2: Local Optimization on All Initializations ---
    
    initializations = [params_hier, params_grid, params_grid5, params_mixed]
    polished = []
    
    for init in initializations:
        if time.time() > deadline - 30:
            break
        polished_init = local_optimize(init, maxiter=1000, ftol=1e-9, gtol=1e-6)
        polished.append(polished_init)
        update_best(polished_init)

    # --- Strategy 3: More Diverse Random Initializations ---
    
    np.random.seed(42)
    for _ in range(30):
        if time.time() > deadline - 50:
            break
        
        # Different random strategies
        strategy = _ % 3
        
        if strategy == 0:
            # Random sizes, random positions
            s_rand = np.random.uniform(0.1, 0.4, n)
            cx_rand = np.random.uniform(0, 1, n)
            cy_rand = np.random.uniform(0, 1, n)
        elif strategy == 1:
            # Greedy placement: place squares one by one
            s_rand = np.random.uniform(0.15, 0.35, n)
            s_rand = np.sort(s_rand)[::-1]  # Largest first
            cx_rand = np.zeros(n)
            cy_rand = np.zeros(n)
            
            placed = []
            for i in range(n):
                s_i = s_rand[i]
                # Try to find a position
                found = False
                for attempt in range(100):
                    cx_try = np.random.uniform(s_i/2, 1 - s_i/2)
                    cy_try = np.random.uniform(s_i/2, 1 - s_i/2)
                    
                    # Check overlap with placed squares
                    valid = True
                    for j in range(len(placed)):
                        cx_j, cy_j, s_j = placed[j]
                        t = (s_i + s_j) / 2.0 - 1e-7
                        if abs(cx_try - cx_j) < t and abs(cy_try - cy_j) < t:
                            valid = False
                            break
                    
                    if valid:
                        cx_rand[i] = cx_try
                        cy_rand[i] = cy_try
                        placed.append((cx_try, cy_try, s_i))
                        found = True
                        break
                
                if not found:
                    # Fallback: place at a corner
                    corners = [(s_i/2, s_i/2), (1-s_i/2, s_i/2), (s_i/2, 1-s_i/2), (1-s_i/2, 1-s_i/2)]
                    cx_rand[i], cy_rand[i] = corners[i % 4]
        else:
            # Grid-based with perturbation
            k = 4
            side = 1.0/k
            idx = np.arange(n)
            cols = idx % k
            rows = idx // k
            cx_rand = (cols + 0.5) * side + np.random.uniform(-0.02, 0.02, n)
            cy_rand = (rows + 0.5) * side + np.random.uniform(-0.02, 0.02, n)
            s_rand = np.full(n, side) + np.random.uniform(-0.02, 0.02, n)
            s_rand = np.clip(s_rand, 0.1, 0.4)
        
        params_rand = np.concatenate([s_rand, cx_rand, cy_rand])
        
        # Quick local optimization
        opt_params = local_optimize(params_rand, maxiter=300, ftol=1e-8, gtol=1e-5)
        update_best(opt_params)
        polished.append(opt_params)

    # --- Strategy 4: Global Optimization (Differential Evolution) ---
    if time.time() < deadline - 60:
        def de_objective(params):
            s = params[:n]
            cx = params[n:2*n]
            cy = params[2*n:3*n]
            
            # Check validity and return -sum(s) + penalty
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
                maxiter=80, 
                popsize=20, 
                tol=1e-5,
                seed=42,
                polish=False
            )
            de_params = res.x
            update_best(de_params)
            
            # Polish DE result
            if de_params is not None:
                polished_de = local_optimize(de_params, maxiter=1500, ftol=1e-10, gtol=1e-7)
                update_best(polished_de)
                polished.append(polished_de)
        except:
            pass

    # --- Strategy 5: High-Precision Polish on Best Candidates ---
    if best_params is not None:
        # Polish the best candidate with coordinate descent
        if time.time() < deadline - 20:
            polished_cd = coordinate_descent(best_params, max_iter=2000, tol=1e-11)
            update_best(polished_cd)
        
        # Final L-BFGS-B with very tight tolerance
        if time.time() < deadline - 10:
            final_polish = local_optimize(best_params, maxiter=3000, ftol=1e-12, gtol=1e-8)
            update_best(final_polish)

    # Also polish other good candidates
    if time.time() < deadline - 15:
        for p in polished[:5]:
            if time.time() > deadline - 10:
                break
            if check_validity(p[:n], p[n:2*n], p[2*n:3*n]):
                score = np.sum(p[:n])
                if score > best_score * 0.95:  # Only polish if close to best
                    polished_p = local_optimize(p, maxiter=1000, ftol=1e-11, gtol=1e-7)
                    update_best(polished_p)

    # Fallback
    if best_params is None:
        centers = np.stack([cx_grid, cy_grid], axis=1).astype(np.float64)
        sides = s_grid.astype(np.float64)
        return centers, sides

    centers = np.stack([best_params[n:2*n], best_params[2*n:3*n]], axis=1)
    sides = best_params[:n]
    
    return centers, sides
