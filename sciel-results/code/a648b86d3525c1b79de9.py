"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START
def construct():
    best_score = -np.inf
    best_result = None
    
    # Number of restarts for multistart
    num_restarts = 40
    
    # Objective function: minimize -sum(radii)
    def objective(vars):
        # vars shape (N, 3): x, y, r
        # Unpack
        r = vars[2::3] # Radii are at indices 2, 5, 8...
        # Maximize sum of radii -> Minimize negative sum
        return -np.sum(r)
        
    # Constraints
    # We define a single function that returns all inequality constraints
    # 1. Boundary constraints:
    #    x_i - r_i >= 0
    #    1 - x_i - r_i >= 0
    #    y_i - r_i >= 0
    #    1 - y_i - r_i >= 0
    # 2. Non-overlap:
    #    (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    
    def all_constraints(vars):
        # Reshape to (N, 3) for easier indexing
        v = vars.reshape(N, 3)
        x = v[:, 0]
        y = v[:, 1]
        r = v[:, 2]
        
        # Boundary constraints (4*N values)
        c_bounds = np.concatenate([
            x - r,
            1 - x - r,
            y - r,
            1 - y - r
        ])
        
        # Pair constraints (N*(N-1)/2 values)
        n_pairs = (N * (N - 1)) // 2
        c_pairs = np.zeros(n_pairs)
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                sum_r = r[i] + r[j]
                c_pairs[idx] = dist_sq - sum_r * sum_r
                idx += 1
        
        return np.concatenate([c_bounds, c_pairs])
    
    constraints = [{'type': 'ineq', 'fun': all_constraints}]
    
    # Bounds:
    # x, y in [0, 1]
    # r in [0, 0.5]
    bounds = [(0, 1), (0, 1), (0, 0.5)] * N
    
    # Initial guesses
    for restart in range(num_restarts):
        # Random initialization with feasible radii
        np.random.seed(restart * 1000 + 42)
        x_init = np.random.uniform(0.1, 0.9, N)
        y_init = np.random.uniform(0.1, 0.9, N)
        
        # Calculate initial radii to be feasible
        r_init = np.zeros(N)
        for i in range(N):
            r_bound = min(x_init[i], 1 - x_init[i], y_init[i], 1 - y_init[i])
            r_min_neighbor = r_bound
            for j in range(N):
                if i != j:
                    dist = np.sqrt((x_init[i] - x_init[j])**2 + (y_init[i] - y_init[j])**2)
                    r_min_neighbor = min(r_min_neighbor, dist / 2.0)
            r_init[i] = r_min_neighbor * 0.95 # Small margin
            
        x0 = np.column_stack((x_init, y_init, r_init)).flatten()
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           constraints=constraints, options={'maxiter': 150, 'ftol': 1e-7})
            
            # Check if result is valid and better
            if res.x is not None:
                # Verify constraints are satisfied (SLSQP might slightly violate)
                # We rely on SLSQP's internal check, but we can also check score
                current_score = -res.fun
                # Ensure radii are positive and within bounds
                r_res = res.x[2::3]
                if np.all(r_res >= 0) and np.all(r_res <= 0.5):
                    if current_score > best_score:
                        best_score = current_score
                        best_result = res.x.reshape(N, 3).copy()
                        
        except Exception as e:
            # If optimization fails, continue to next restart
            continue
            
    return best_result
# EVOLVE-BLOCK-END
