# EVOLVE-BLOCK-START
import numpy as np
import time

def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares inside the unit square [0,1]^2 with pairwise
    disjoint interiors to maximize the sum of their side lengths.

    Strategy:
      1. Build a set of diverse initial configurations (grid, hand-crafted, random).
      2. For each, run a smooth constrained optimization (SLSQP) that maximizes
         sum(sides) subject to non-overlap and boundary constraints.
      3. Polish the best result with a high-precision bisection pass that extracts
         additional side length from near-optimal configurations.
      4. Return the best valid configuration found within the time budget.
    """
    n = 10
    tol = 1e-7
    deadline = time.time() + 600.0  # safety margin before 630s kill

    def check_valid(centers, sides, tol=1e-7):
        """Check if configuration is valid: all squares inside [0,1]^2, no overlaps."""
        if centers.shape != (n, 2) or sides.shape != (n,):
            return False
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if not np.all(sides > 0):
            return False
        # Boundary check
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s/2 < -tol or cx + s/2 > 1 + tol:
                return False
            if cy - s/2 < -tol or cy + s/2 > 1 + tol:
                return False
        # Overlap check
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                if abs(centers[i,0] - centers[j,0]) < t and abs(centers[i,1] - centers[j,1]) < t:
                    return False
        return True

    def objective(x):
        """x = [cx_0, cy_0, s_0, cx_1, cy_1, s_1, ...]"""
        centers = x[0::3].reshape(-1, 2)  # wrong, let's use different layout
        # Actually, let's use x = [cx_0..cx_9, cy_0..cy_9, s_0..s_9]
        return -np.sum(x[2*n:3*n])  # minimize negative sum of sides

    def objective_and_grad(x):
        sides = x[2*n:3*n]
        return -np.sum(sides), -np.concatenate([np.zeros(2*n), np.ones(n)])

    def constraints(x):
        """Return list of constraint dicts for SLSQP.
        Each constraint: {'type': 'ineq', 'fun': f} where f >= 0 means feasible.
        """
        cx = x[0:n]
        cy = x[n:2*n]
        s = x[2*n:3*n]
        cons = []
        # Boundary constraints: cx >= s/2, cx <= 1 - s/2, etc.
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - x[2*n+i]/2})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[i] - x[2*n+i]/2})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[n+i] - x[2*n+i]/2})
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[n+i] - x[2*n+i]/2})
        # Non-overlap constraints: for each pair (i,j), at least one of:
        #   cx_i - s_i/2 >= cx_j + s_j/2  (i is to the right of j)
        #   cx_j - s_j/2 >= cx_i + s_i/2  (j is to the right of i)
        #   cy_i - s_i/2 >= cy_j + s_j/2  (i is above j)
        #   cy_j - s_j/2 >= cy_i + s_i/2  (j is above i)
        # We use a smooth approximation: max of the four separations should be >= 0.
        # For SLSQP, we can use the minimum of the four as a single constraint.
        for i in range(n):
            for j in range(i+1, n):
                def sep(x, i=i, j=j):
                    cx_i, cx_j = x[i], x[j]
                    cy_i, cy_j = x[n+i], x[n+j]
                    s_i, s_j = x[2*n+i], x[2*n+j]
                    # Separation in x: min distance to separate in x
                    sep_x1 = cx_i - s_i/2 - (cx_j + s_j/2)  # i right of j
                    sep_x2 = cx_j - s_j/2 - (cx_i + s_i/2)  # j right of i
                    sep_y1 = cy_i - s_i/2 - (cy_j + s_j/2)  # i above j
                    sep_y2 = cy_j - s_j/2 - (cy_i + s_i/2)  # j above i
                    # We need at least one of these to be >= 0.
                    # Use a soft-min approximation: log-sum-exp trick
                    # Actually, for SLSQP, let's just use the min directly.
                    return min(sep_x1, sep_x2, sep_y1, sep_y2)
                cons.append({'type': 'ineq', 'fun': sep})
        return cons

    def generate_initial_configs():
        """Generate diverse initial configurations."""
        configs = []
        
        # 1. Grid configuration (baseline)
        k = int(np.ceil(np.sqrt(n)))
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        x0 = np.concatenate([cx, cy, np.full(n, side)])
        configs.append(x0)
        
        # 2. Hand-crafted: one large square in corner, smaller ones around
        # Large square: side 0.5, at (0.25, 0.25)
        # 4 medium squares: side 0.25, filling the rest
        # 5 small squares: side 0.125
        cx = np.array([0.25, 0.75, 0.75, 0.25, 0.75, 0.5, 0.75, 0.625, 0.875, 0.5])
        cy = np.array([0.25, 0.25, 0.75, 0.75, 0.5, 0.75, 0.625, 0.875, 0.5, 0.375])
        s = np.array([0.5, 0.25, 0.25, 0.25, 0.25, 0.125, 0.125, 0.125, 0.125, 0.125])
        x0 = np.concatenate([cx, cy, s])
        configs.append(x0)
        
        # 3. Random configurations
        np.random.seed(42)
        for trial in range(5):
            cx = np.random.uniform(0.1, 0.9, n)
            cy = np.random.uniform(0.1, 0.9, n)
            s = np.random.uniform(0.05, 0.3, n)
            x0 = np.concatenate([cx, cy, s])
            configs.append(x0)
        
        return configs

    def polish(centers, sides, tol=1e-9, max_iter=80):
        """Polish a valid configuration by trying to increase side lengths."""
        centers = centers.copy()
        sides = sides.copy()
        
        for _ in range(max_iter):
            improved = False
            for i in range(n):
                # Try to increase side i by a small amount
                best_delta = 0
                lo, hi = 0.0, 0.05
                for _ in range(max_iter):
                    mid = (lo + hi) / 2
                    new_s = sides[i] + mid
                    if new_s <= 0:
                        hi = mid
                        continue
                    # Check if increasing side i by mid is valid
                    test_centers = centers.copy()
                    test_sides = sides.copy()
                    test_sides[i] = new_s
                    if check_valid(test_centers, test_sides, tol=1e-10):
                        lo = mid
                        best_delta = mid
                        improved = True
                    else:
                        hi = mid
                if best_delta > 0:
                    sides[i] += best_delta
            if not improved:
                break
        return centers, sides

    # Main optimization loop
    best_score = -np.inf
    best_result = None

    configs = generate_initial_configs()

    for x0 in configs:
        if time.time() > deadline - 60:
            break
        
        try:
            from scipy.optimize import minimize
            
            # Bounds: cx, cy in [0, 1], s in [0.01, 1.0]
            bounds = []
            for i in range(n):
                bounds.append((0.0, 1.0))  # cx
            for i in range(n):
                bounds.append((0.0, 1.0))  # cy
            for i in range(n):
                bounds.append((0.01, 1.0))  # s
            
            cons = constraints(x0)
            
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            
            if result.success or result.fun < best_score:
                x = result.x
                cx = x[0:n]
                cy = x[n:2*n]
                s = x[2*n:3*n]
                centers = np.stack([cx, cy], axis=1)
                sides = s
                
                if check_valid(centers, sides, tol=1e-7):
                    score = np.sum(sides)
                    if score > best_score:
                        best_score = score
                        best_result = (centers, sides)
        except Exception:
            continue
    
    # Polish the best result
    if best_result is not None:
        centers, sides = best_result
        if time.time() < deadline - 30:
            centers, sides = polish(centers, sides)
            if check_valid(centers, sides, tol=1e-7):
                best_result = (centers, sides)
    
    # Fallback to grid if optimization failed
    if best_result is None:
        k = int(np.ceil(np.sqrt(n)))
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)
        best_result = (centers, sides)
    
    return best_result

# EVOLVE-BLOCK-END
