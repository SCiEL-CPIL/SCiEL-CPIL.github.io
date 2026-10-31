"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 26

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s limit
    
    def elapsed():
        return time.time() - start_time
    
    # Initialize with a grid layout
    side = int(np.ceil(np.sqrt(N)))  # 6 for N=26
    # Create grid positions
    grid_positions = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side
            grid_positions.append([x, y, r])
    
    # Take first N positions
    init = np.array(grid_positions[:N])
    
    # Flatten to 1D vector: [x1,y1,r1, x2,y2,r2, ...]
    def to_vec(arr):
        return arr.flatten()
    
    def from_vec(v):
        return v.reshape(-1, 3)
    
    x0 = to_vec(init)
    
    # Penalty-based objective
    def penalty_objective(v, penalty_weight=100.0):
        arr = from_vec(v)
        n = arr.shape[0]
        
        # Sum of radii (to maximize, so minimize negative)
        sum_r = np.sum(arr[:, 2])
        
        # Constraint violations
        penalty = 0.0
        
        # Boundary constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
        for i in range(n):
            x, y, r = arr[i]
            # r must be non-negative
            if r < 0:
                penalty += penalty_weight * r * r
            # Inside square
            if x - r < 0:
                penalty += penalty_weight * (x - r) * (x - r)
            if 1 - x - r < 0:
                penalty += penalty_weight * (1 - x - r) * (1 - x - r)
            if y - r < 0:
                penalty += penalty_weight * (y - r) * (y - r)
            if 1 - y - r < 0:
                penalty += penalty_weight * (1 - y - r) * (1 - y - r)
        
        # Non-overlapping constraints
        for i in range(n):
            for j in range(i + 1, n):
                xi, yi, ri = arr[i]
                xj, yj, rj = arr[j]
                dist = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                min_dist = ri + rj
                if dist < min_dist:
                    violation = min_dist - dist
                    penalty += penalty_weight * violation * violation
        
        return -sum_r + penalty
    
    # Bounds: x in [0,1], y in [0,1], r in [0, 0.5]
    bounds = []
    for _ in range(N):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((0.0, 0.5))  # r
    
    # Run optimization with penalty method
    best_result = None
    best_obj = float('inf')
    
    # Multiple restarts with different initializations and penalty weights
    penalty_weights = [50.0, 100.0, 200.0, 500.0]
    
    for pw in penalty_weights:
        if elapsed() > deadline - 30:
            break
            
        # Try with the grid init
        result = minimize(
            penalty_objective,
            x0,
            args=(pw,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if result.success or result.fun < best_obj:
            best_obj = result.fun
            best_result = result.x
        
        if elapsed() > deadline - 20:
            break
    
    # Convert best result back
    if best_result is not None:
        final_arr = from_vec(best_result)
    else:
        final_arr = init
    
    # Polish with SLSQP using hard constraints
    if elapsed() < deadline - 5:
        def hard_objective(v):
            arr = from_vec(v)
            return -np.sum(arr[:, 2])
        
        # Equality/inequality constraints for SLSQP
        constraints = []
        
        # Boundary constraints
        for i in range(N):
            # r_i <= x_i
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i: v[3*i] - v[3*i+2]
            })
            # r_i <= 1 - x_i
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]
            })
            # r_i <= y_i
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]
            })
            # r_i <= 1 - y_i
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]
            })
        
        # Non-overlapping constraints
        for i in range(N):
            for j in range(i + 1, N):
                def make_constraint(i, j):
                    def fun(v):
                        xi, yi, ri = v[3*i], v[3*i+1], v[3*i+2]
                        xj, yj, rj = v[3*j], v[3*j+1], v[3*j+2]
                        dist_sq = (xi - xj) ** 2 + (yi - yj) ** 2
                        min_dist_sq = (ri + rj) ** 2
                        return dist_sq - min_dist_sq
                    return fun
                constraints.append({
                    'type': 'ineq',
                    'fun': make_constraint(i, j)
                })
        
        # Try SLSQP polish
        try:
            result2 = minimize(
                hard_objective,
                best_result if best_result is not None else x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            
            if result2.fun < best_obj:
                best_result = result2.x
                best_obj = result2.fun
                final_arr = from_vec(best_result)
        except Exception:
            pass
    
    # Final validation and cleanup
    final_arr = np.clip(final_arr, 0.0, 1.0)
    # Ensure radii are non-negative and circles are inside
    for i in range(N):
        x, y, r = final_arr[i]
        r = min(r, x, 1 - x, y, 1 - y)
        r = max(0, r)
        final_arr[i] = [x, y, r]
    
    return final_arr
# EVOLVE-BLOCK-END
