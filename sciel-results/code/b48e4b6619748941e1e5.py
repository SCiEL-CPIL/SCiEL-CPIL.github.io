import numpy as np
from scipy.optimize import minimize
import itertools

def construct():
    # 1. Generate initial points using the hexagonal lattice logic from the seed
    # This provides a high-quality starting point for the optimizer.
    
    points = [(0, 0)]
    dirs = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    for d in dirs:
        points.append((d[0], d[1]))
        
    ring2_points = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if max(abs(q), abs(r), abs(q+r)) == 2:
                ring2_points.append((q, r))
                
    sqrt3 = np.sqrt(3)
    cart_points = []
    cart_points.append((0.0, 0.0))
    for q, r in dirs:
        x = q + r/2.0
        y = sqrt3/2.0 * r
        cart_points.append((x, y))
    for q, r in ring2_points:
        x = q + r/2.0
        y = sqrt3/2.0 * r
        cart_points.append((x, y))
        
    outer_indices = list(range(7, 19))
    best_min_max_dist = float('inf')
    best_subset = None
    
    for remove_combo in itertools.combinations(outer_indices, 3):
        remove_set = set(remove_combo)
        current_points = [cart_points[i] for i in range(19) if i not in remove_set]
        
        max_dist = 0.0
        n = len(current_points)
        for i in range(n):
            for j in range(i+1, n):
                dx = current_points[i][0] - current_points[j][0]
                dy = current_points[i][1] - current_points[j][1]
                dist_sq = dx*dx + dy*dy
                if dist_sq > max_dist:
                    max_dist = dist_sq
        max_dist = np.sqrt(max_dist)
        
        if max_dist < best_min_max_dist:
            best_min_max_dist = max_dist
            best_subset = current_points
            
    x0 = np.array(best_subset, dtype=float)
    
    # 2. Scale x0 so that the minimum pairwise distance is 1.0
    # This fixes the scale, allowing us to minimize the maximum distance.
    n = len(x0)
    min_d = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(x0[i] - x0[j])
            if d < min_d:
                min_d = d
                
    if min_d > 0:
        x0 = x0 / min_d
    else:
        # Fallback if something went wrong (should not happen)
        return x0

    # 3. Define Objective and Constraints for Optimization
    # Objective: Minimize maximum pairwise distance
    # Constraints: All pairwise distances >= 1.0
    
    def objective(x):
        pts = x.reshape(n, 2)
        max_d = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                if d > max_d:
                    max_d = d
        return max_d

    # Create constraints for SLSQP
    # d_ij >= 1  =>  d_ij - 1 >= 0
    constraints = []
    for i in range(n):
        for j in range(i+1, n):
            def make_constraint(i, j):
                def constraint(x):
                    pts = x.reshape(n, 2)
                    d = np.linalg.norm(pts[i] - pts[j])
                    return d - 1.0
                return constraint
            constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})
            
    # 4. Run Optimization
    # Use SLSQP as it handles inequalities well.
    # Set maxiter to prevent timeout.
    try:
        res = minimize(objective, x0.flatten(), method='SLSQP', constraints=constraints, 
                       options={'maxiter': 200, 'ftol': 1e-6})
        
        if res.success:
            return res.x.reshape(n, 2)
        else:
            # If optimization fails, return the scaled initial guess
            return x0
    except Exception:
        return x0
