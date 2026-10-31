import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 26

def construct():
    # Deadline guard: 150s limit, aim to finish by 140s
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Objective: Minimize negative sum of radii
    def objective(x):
        # x is [x1, y1, r1, x2, y2, r2, ..., xN, yN, rN]
        radii = x[2::3]
        return -np.sum(radii)
    
    # Constraints:
    # 1. Boundary constraints: x >= r, y >= r, 1-x >= r, 1-y >= r
    # 2. Pairwise distance: dist^2 >= (ri + rj)^2
    
    def make_constraints_func():
        cons = []
        
        # Boundary constraints
        # x_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda x: x[0::3] - x[2::3]})
        # y_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda x: x[1::3] - x[2::3]})
        # 1 - x_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda x: 1 - x[0::3] - x[2::3]})
        # 1 - y_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda x: 1 - x[1::3] - x[2::3]})
        
        # Pairwise constraints
        # We define a single function that returns an array of values >= 0
        # However, SLSQP 'ineq' fun can return an array.
        # Let's create a function that computes all pairwise constraints.
        
        def pairwise_constraints(x):
            # Extract positions and radii
            xs = x[0::3]
            ys = x[1::3]
            rs = x[2::3]
            
            # Compute squared distances between all pairs
            # We need i < j
            constraints_vals = []
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist_sq = dx*dx + dy*dy
                    sum_r = rs[i] + rs[j]
                    # Constraint: dist_sq - sum_r^2 >= 0
                    constraints_vals.append(dist_sq - sum_r*sum_r)
            return np.array(constraints_vals)
        
        cons.append({'type': 'ineq', 'fun': pairwise_constraints})
        
        return cons

    constraints = make_constraints_func()
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r

    # Initializers
    initializers = []
    
    # 1. 5x5 Grid + 1 Center
    # 5x5 grid of circles.
    # To make it feasible, we start with small radii and let optimizer grow them.
    # Or start with a feasible configuration.
    # Let's create a 5x5 grid where circles touch boundaries and neighbors.
    # Side length 1, 5 circles. Radius ~ 0.1.
    # Center coords: 0.1, 0.3, 0.5, 0.7, 0.9
    grid_coords = []
    for i in range(5):
        for j in range(5):
            x = 0.1 + 0.2 * i
            y = 0.1 + 0.2 * j
            grid_coords.append((x, y))
    
    # Add 26th circle in the center (0.5, 0.5) with small radius
    grid_coords.append((0.5, 0.5))
    
    # Construct x vector for grid
    x0_grid = np.zeros(3 * N)
    for i, (cx, cy) in enumerate(grid_coords):
        x0_grid[3*i] = cx
        x0_grid[3*i+1] = cy
        # Start with small radius to ensure feasibility
        x0_grid[3*i+2] = 0.05 
        
    # Check if grid initializer is feasible
    # The 5x5 grid with r=0.05 is feasible (dist between centers 0.2, sum r 0.1).
    # Center circle at (0.5, 0.5) is at distance 0.2 from nearest grid centers.
    # Sum r = 0.1. Feasible.
    initializers.append(x0_grid)
    
    # 2. Jittered Grid
    rng = np.random.default_rng(42)
    x0_jitter = x0_grid.copy()
    # Add small jitter to positions
    x0_jitter[0::3] += rng.uniform(-0.02, 0.02, N)
    x0_jitter[1::3] += rng.uniform(-0.02, 0.02, N)
    # Clip to ensure within bounds roughly, though optimizer will fix
    x0_jitter[0::3] = np.clip(x0_jitter[0::3], 0.05, 0.95)
    x0_jitter[1::3] = np.clip(x0_jitter[1::3], 0.05, 0.95)
    initializers.append(x0_jitter)
    
    # 3. Random Feasible
    # Generate random points and small radii
    for k in range(5):
        x0_rand = np.zeros(3 * N)
        # Random positions in [0.1, 0.9]
        x0_rand[0::3] = rng.uniform(0.1, 0.9, N)
        x0_rand[1::3] = rng.uniform(0.1, 0.9, N)
        x0_rand[2::3] = 0.02 # Small radius
        initializers.append(x0_rand)
        
    best_x = None
    best_val = -np.inf
    
    # Optimization Loop
    # We will try each initializer. If time allows, we can do more random restarts.
    # SLSQP can be slow with many constraints. 26 circles -> 325 pairwise constraints.
    # This might be heavy. We should limit the number of iterations per run or use a tighter tol.
    
    for x0 in initializers:
        if time.time() > deadline - 10: # Stop if less than 10s left
            break
            
        try:
            # SLSQP with moderate tolerances for speed
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           constraints=constraints, 
                           options={'ftol': 1e-6, 'maxiter': 200, 'disp': False})
            
            if res.success or res.status == 0:
                x_opt = res.x
                # Validate manually
                # Check bounds
                if np.any(x_opt[0::3] < 0) or np.any(x_opt[0::3] > 1) or \
                   np.any(x_opt[1::3] < 0) or np.any(x_opt[1::3] > 1) or \
                   np.any(x_opt[2::3] < 0) or np.any(x_opt[2::3] > 0.5):
                    continue
                    
                # Check pairwise
                xs = x_opt[0::3]
                ys = x_opt[1::3]
                rs = x_opt[2::3]
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        dist_sq = (xs[i]-xs[j])**2 + (ys[i]-ys[j])**2
                        sum_r = rs[i] + rs[j]
                        if dist_sq < sum_r*sum_r - 1e-8: # Tolerance
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                    
                val = np.sum(rs)
                if val > best_val:
                    best_val = val
                    best_x = x_opt.copy()
                    
        except Exception:
            pass
            
    # Polish Phase
    # If we have a good solution and time remains, run a high-precision polish
    if best_x is not None and time.time() < deadline - 5:
        try:
            # Tighter tolerances
            res_polish = minimize(objective, best_x, method='SLSQP', bounds=bounds, 
                           constraints=constraints, 
                           options={'ftol': 1e-9, 'maxiter': 100, 'disp': False})
            
            if res_polish.success or res_polish.status == 0:
                x_polish = res_polish.x
                # Quick validation
                xs = x_polish[0::3]
                ys = x_polish[1::3]
                rs = x_polish[2::3]
                valid = True
                # Check bounds
                if np.any(rs < 0) or np.any(rs > 0.5):
                    valid = False
                else:
                    for i in range(N):
                        if xs[i] < rs[i] - 1e-8 or xs[i] + rs[i] > 1 + 1e-8:
                            valid = False
                            break
                        if ys[i] < rs[i] - 1e-8 or ys[i] + rs[i] > 1 + 1e-8:
                            valid = False
                            break
                if valid:
                    # Check pairwise
                    for i in range(N):
                        for j in range(i+1, N):
                            dist_sq = (xs[i]-xs[j])**2 + (ys[i]-ys[j])**2
                            sum_r = rs[i] + rs[j]
                            if dist_sq < sum_r*sum_r - 1e-8:
                                valid = False
                                break
                        if not valid:
                            break
                            
                if valid:
                    val_polish = np.sum(rs)
                    if val_polish > best_val:
                        best_val = val_polish
                        best_x = x_polish.copy()
                        
        except Exception:
            pass

    # If no valid solution was found (should not happen with grid start), return trivial grid
    if best_x is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Format output: (N, 3) array
    # best_x is [x1, y1, r1, x2, y2, r2, ...]
    result = best_x.reshape(N, 3)
    
    # Final sanity check and cleanup
    # Ensure radii are positive
    result[:, 2] = np.maximum(result[:, 2], 0)
    
    return result
