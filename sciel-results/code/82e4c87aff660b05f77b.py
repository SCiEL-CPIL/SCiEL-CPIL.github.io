import numpy as np
from scipy.optimize import minimize
import time

N = 26

def construct():
    start_time = time.time()
    max_time = 110.0  # Safety margin for 120s limit, leave time for repair/output

    best_result = None
    best_score = -np.inf

    # Define constraints and objective
    
    def objective(params):
        # params: [x1, y1, r1, x2, y2, r2, ...]
        radii = params[2::3]
        return -np.sum(radii)

    def boundary_constraints(params):
        vals = np.zeros(4*N)
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            r = params[3*i+2]
            vals[4*i] = x - r
            vals[4*i+1] = 1 - r - x
            vals[4*i+2] = y - r
            vals[4*i+3] = 1 - r - y
        return vals

    def overlap_constraints(params):
        vals = np.zeros(N*(N-1)//2)
        idx = 0
        for i in range(N):
            xi = params[3*i]
            yi = params[3*i+1]
            ri = params[3*i+2]
            for j in range(i+1, N):
                xj = params[3*j]
                yj = params[3*j+1]
                rj = params[3*j+2]
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                sum_r = ri + rj
                vals[idx] = dist_sq - sum_r**2
                idx += 1
        return vals

    def make_constraint_list():
        cons = []
        cons.append({
            'type': 'ineq',
            'fun': boundary_constraints
        })
        cons.append({
            'type': 'ineq',
            'fun': overlap_constraints
        })
        return cons

    constraints = make_constraint_list()
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((1e-6, 0.5)) # r

    # Helper to check validity with strict tolerance
    def is_valid(params, tol=1e-8):
        for i in range(N):
            x, y, r = params[3*i], params[3*i+1], params[3*i+2]
            if r < tol:
                return False
            if x < r - tol or x > 1 - r + tol:
                return False
            if y < r - tol or y > 1 - r + tol:
                return False
        for i in range(N):
            xi, yi, ri = params[3*i], params[3*i+1], params[3*i+2]
            for j in range(i+1, N):
                xj, yj, rj = params[3*j], params[3*j+1], params[3*j+2]
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                sum_r = ri + rj
                if dist_sq < sum_r**2 - tol:
                    return False
        return True

    # Helper to repair a solution by shrinking radii if invalid
    def repair_solution(params):
        params = params.copy()
        # Iteratively shrink radii if constraints are violated
        # We shrink all radii slightly to maintain relative packing
        for _ in range(50):
            if is_valid(params):
                break
            
            # Check which constraint is most violated? 
            # Simple approach: shrink all radii by 0.5%
            # This is safe but might lose some score.
            # Better: identify violating pairs and shrink those specific radii?
            # For robustness, global shrink is safer against oscillation.
            for i in range(N):
                params[3*i+2] *= 0.995
                
        # Final clip to ensure absolute bounds are respected
        for i in range(N):
            x, y, r = params[3*i], params[3*i+1], params[3*i+2]
            r = min(r, x, 1-x, y, 1-y)
            if r < 1e-6:
                r = 1e-6
                # Adjust x,y to center if r was too big
                x = 0.5
                y = 0.5
            params[3*i] = x
            params[3*i+1] = y
            params[3*i+2] = r
            
        # One last check, if still invalid, return a trivial valid solution
        if not is_valid(params):
            return None
        return params

    # Initializations
    inits = []
    
    # 1. Grid with small initial radii to ensure feasibility
    rows = 5
    cols = 6
    xs = np.linspace(0.5/cols, 1-0.5/cols, cols)
    ys = np.linspace(0.5/rows, 1-0.5/rows, rows)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()
    
    # Calculate max radius that fits in grid cell without overlap
    # Cell width = 1/cols, Cell height = 1/rows
    # Max r = min(1/(2*cols), 1/(2*rows)) * 0.9
    r_init = min(1.0/(2.0*cols), 1.0/(2.0*rows)) * 0.8
    
    init_params = []
    for i in range(N):
        x = grid_x[i]
        y = grid_y[i]
        r = r_init
        # Ensure within bounds
        x = np.clip(x, r, 1-r)
        y = np.clip(y, r, 1-r)
        init_params.extend([x, y, r])
    inits.append(np.array(init_params))
    
    # 2. Random initialization with small radii
    x_rand = np.random.uniform(0.2, 0.8, N)
    y_rand = np.random.uniform(0.2, 0.8, N)
    r_rand = np.full(N, 0.05)
    init_rand = []
    for i in range(N):
        init_rand.extend([x_rand[i], y_rand[i], r_rand[i]])
    inits.append(np.array(init_rand))

    # Run optimization
    for init in inits:
        if time.time() - start_time > max_time:
            break
            
        try:
            res = minimize(
                objective, 
                init, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            
            # Check if the result is valid
            # SLSQP might report success but still be slightly infeasible due to numerical issues
            # Or it might fail but land in a good spot.
            # We always try to repair and validate.
            
            candidate = res.x.copy()
            repaired = repair_solution(candidate)
            
            if repaired is not None:
                score = -np.sum(repaired[2::3])
                if score > best_score:
                    best_score = score
                    best_result = repaired
                    
        except Exception as e:
            pass

    # If no valid result found, return a trivial valid one
    if best_result is None:
        # Fallback: small circles in a grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.4 / side
                cells.append([x, y, r])
        best_result = np.array(cells[:N])
        
    # Format output
    result_array = np.zeros((N, 3))
    for i in range(N):
        result_array[i] = best_result[3*i:3*i+3]
        
    return result_array
