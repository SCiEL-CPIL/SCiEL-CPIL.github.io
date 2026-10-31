"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 26

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    # Initial guess: grid-based placement
    side = int(np.ceil(np.sqrt(N)))
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.001
                positions.append([x, y, r])
    
    x0 = np.array(positions, dtype=float)
    n = N
    
    # Objective: maximize sum of radii -> minimize negative sum of radii
    def objective(x):
        return -np.sum(x[2::3])
    
    # Constraints:
    # 1. Non-overlapping: for each pair (i,j), (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
    # 2. Inside unit square: xi >= ri, xi <= 1-ri, yi >= ri, yi <= 1-ri
    
    def make_constraint(i, j):
        def constraint(x):
            dx = x[3*i] - x[3*j]
            dy = x[3*i+1] - x[3*j+1]
            ri = x[3*i+2]
            rj = x[3*j+2]
            return dx*dx + dy*dy - (ri + rj)**2
        constraint.name = f"nonoverlap_{i}_{j}"
        return constraint
    
    def make_boundary(k, coord_idx, sign):
        # sign=1: x >= r  -> x - r >= 0
        # sign=-1: 1 - x - r >= 0
        def constraint(x):
            if sign == 1:
                return x[3*k + coord_idx] - x[3*k + 2]
            else:
                return 1.0 - x[3*k + coord_idx] - x[3*k + 2]
        constraint.name = f"boundary_{k}_{coord_idx}_{sign}"
        return constraint
    
    constraints = []
    
    # Non-overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})
    
    # Boundary constraints
    for k in range(n):
        constraints.append({'type': 'ineq', 'fun': make_boundary(k, 0, 1)})   # x >= r
        constraints.append({'type': 'ineq', 'fun': make_boundary(k, 0, -1)})  # 1-x >= r
        constraints.append({'type': 'ineq', 'fun': make_boundary(k, 1, 1)})   # y >= r
        constraints.append({'type': 'ineq', 'fun': make_boundary(k, 1, -1)})  # 1-y >= r
    
    # Radius positivity
    for k in range(n):
        def make_radius(k):
            def constraint(x):
                return x[3*k+2]
            constraint.name = f"radius_{k}"
            return constraint
        constraints.append({'type': 'ineq', 'fun': make_radius(k)})
    
    bounds = [(0, 1)] * (3 * n)
    
    best_x = x0.copy()
    best_val = objective(x0)
    
    # Try multiple initializations and local optimizations
    for trial in range(5):
        if time.time() > deadline - 30:
            break
        
        # Perturb initial guess
        if trial > 0:
            x0_trial = x0 + np.random.randn(3*n) * 0.01
            # Clip to valid range
            x0_trial[0::3] = np.clip(x0_trial[0::3], 0.01, 0.99)
            x0_trial[1::3] = np.clip(x0_trial[1::3], 0.01, 0.99)
            x0_trial[2::3] = np.clip(x0_trial[2::3], 0.01, 0.49)
        else:
            x0_trial = x0.copy()
        
        try:
            res = minimize(
                objective,
                x0_trial,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-8}
            )
            
            if res.success or res.fun < best_val:
                # Check feasibility
                x_res = res.x
                feasible = True
                for i in range(n):
                    for j in range(i+1, n):
                        dx = x_res[3*i] - x_res[3*j]
                        dy = x_res[3*i+1] - x_res[3*j+1]
                        ri = x_res[3*i+2]
                        rj = x_res[3*j+2]
                        if dx*dx + dy*dy < (ri + rj)**2 - 1e-10:
                            feasible = False
                            break
                    if not feasible:
                        break
                    # Check boundaries
                    if x_res[3*i] < x_res[3*i+2] - 1e-10 or \
                       x_res[3*i] > 1 - x_res[3*i+2] + 1e-10 or \
                       x_res[3*i+1] < x_res[3*i+2] - 1e-10 or \
                       x_res[3*i+1] > 1 - x_res[3*i+2] + 1e-10:
                        feasible = False
                        break
                if feasible:
                    best_x = x_res
                    best_val = res.fun
        except Exception as e:
            pass
    
    # Final result: extract (x, y, r)
    result = np.zeros((n, 3))
    result[:, 0] = best_x[0::3]
    result[:, 1] = best_x[1::3]
    result[:, 2] = best_x[2::3]
    
    return result
# EVOLVE-BLOCK-END
