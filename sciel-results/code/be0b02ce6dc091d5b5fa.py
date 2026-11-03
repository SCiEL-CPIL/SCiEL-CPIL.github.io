import numpy as np
from scipy.optimize import minimize
import time

N = 26

def construct():
    start_time = time.time()
    max_time = 110.0  # Leave 10s margin
    
    best_sum_r = 0
    best_solution = None
    
    # Define objective function: minimize negative sum of radii
    def objective(v):
        # v contains x, y, r for each circle
        # Structure: [x1, y1, r1, x2, y2, r2, ...]
        radii = v[2::3]
        return -np.sum(radii)
    
    # Define constraints
    # Inequality constraints must be >= 0
    def make_constraints():
        cons = []
        
        # Boundary constraints
        for i in range(N):
            xi = 3*i
            yi = 3*i + 1
            ri = 3*i + 2
            
            # x_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]})
            # 1 - (x_i + r_i) >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - (v[3*i] + v[3*i+2])})
            # y_i - r_i >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]})
            # 1 - (y_i + r_i) >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - (v[3*i+1] + v[3*i+2])})
            
        # Non-overlap constraints
        for i in range(N):
            for j in range(i+1, N):
                xi, yi, ri = 3*i, 3*i+1, 3*i+2
                xj, yj, rj = 3*j, 3*j+1, 3*j+2
                
                def dist_constraint(v, i=i, j=j):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    return dist - (v[3*i+2] + v[3*j+2])
                
                cons.append({'type': 'ineq', 'fun': dist_constraint})
        
        return cons
    
    constraints = make_constraints()
    
    # Bounds for variables
    # x, y in [0, 1], r in [0, 0.5] (max possible radius is 0.5)
    bounds = []
    for i in range(N):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0, 0.5)) # r
        
    # Generate multiple initial guesses
    # 1. Grid-like initial guess
    # 2. Random initial guesses
    
    initial_guesses = []
    
    # Grid guess
    side = int(np.ceil(np.sqrt(N)))
    for k in range(10):
        # Perturb grid slightly
        rng = np.random.default_rng(k)
        x_coords = []
        y_coords = []
        r_coords = []
        for i in range(side):
            for j in range(side):
                if len(x_coords) < N:
                    x0 = (i + 0.5) / side
                    y0 = (j + 0.5) / side
                    r0 = 0.5 / side
                    # Add small noise
                    x_coords.append(x0 + rng.uniform(-0.01, 0.01))
                    y_coords.append(y0 + rng.uniform(-0.01, 0.01))
                    r_coords.append(r0)
        
        v0 = np.array([item for sublist in zip(x_coords, y_coords, r_coords) for item in sublist])
        initial_guesses.append(v0)
        
    # Random guesses
    for k in range(20, 40):
        rng = np.random.default_rng(k)
        v0 = np.zeros(N*3)
        for i in range(N):
            v0[3*i] = rng.uniform(0.1, 0.9)
            v0[3*i+1] = rng.uniform(0.1, 0.9)
            v0[3*i+2] = rng.uniform(0.05, 0.2)
        initial_guesses.append(v0)
        
    # Try optimizing from each initial guess
    for v0 in initial_guesses:
        if time.time() - start_time > max_time:
            break
            
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 500, 'ftol': 1e-9})
            if res.success:
                sum_r = -res.fun
                if sum_r > best_sum_r:
                    best_sum_r = sum_r
                    best_solution = res.x
        except Exception:
            pass
            
    # If no solution found, fallback to grid
    if best_solution is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-5])
        best_solution = np.array([item for sublist in cells for item in sublist])
        
    # Format output as (n, 3) array
    result = best_solution.reshape(N, 3)
    return result

