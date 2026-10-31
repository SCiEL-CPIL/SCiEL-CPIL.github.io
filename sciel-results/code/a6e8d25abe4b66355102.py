"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 120.0  # seconds, with safety margin
    
    # We'll use a multi-start local optimization approach
    # Each start: random initialization, then SLSQP with constraints
    # Constraints: 
    #   1. x_i >= r_i, x_i <= 1-r_i, y_i >= r_i, y_i <= 1-r_i (boundary)
    #   2. (x_i-x_j)^2 + (y_i-y_j)^2 >= (r_i+r_j)^2 (non-overlap)
    
    # Variables: [x_0, y_0, r_0, x_1, y_1, r_1, ..., x_25, y_25, r_25]
    # Objective: maximize sum(r_i) => minimize -sum(r_i)
    
    best_result = None
    best_value = -np.inf
    
    def objective(v):
        r = v[2::3]
        return -np.sum(r)
    
    def make_constraints():
        constraints = []
        
        # Boundary constraints: for each circle i
        for i in range(N):
            xi = 3*i
            yi = 3*i + 1
            ri = 3*i + 2
            
            # x_i >= r_i  =>  x_i - r_i >= 0
            constraints.append({'type': 'ineq', 'fun': lambda v, xi=xi, ri=ri: v[xi] - v[ri]})
            # x_i <= 1 - r_i  =>  1 - x_i - r_i >= 0
            constraints.append({'type': 'ineq', 'fun': lambda v, xi=xi, ri=ri: 1.0 - v[xi] - v[ri]})
            # y_i >= r_i  =>  y_i - r_i >= 0
            constraints.append({'type': 'ineq', 'fun': lambda v, yi=yi, ri=ri: v[yi] - v[ri]})
            # y_i <= 1 - r_i  =>  1 - y_i - r_i >= 0
            constraints.append({'type': 'ineq', 'fun': lambda v, yi=yi, ri=ri: 1.0 - v[yi] - v[ri]})
            # r_i > 0  =>  r_i >= 0 (we'll use a small lower bound)
            constraints.append({'type': 'ineq', 'fun': lambda v, ri=ri: v[ri] - 1e-6})
        
        # Non-overlap constraints: for each pair (i, j), i < j
        for i in range(N):
            for j in range(i+1, N):
                xi = 3*i
                yi = 3*i + 1
                ri = 3*i + 2
                xj = 3*j
                yj = 3*j + 1
                rj = 3*j + 2
                
                # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda v, xi=xi, yi=yi, ri=ri, xj=xj, yj=yj, rj=rj: 
                        (v[xi] - v[xj])**2 + (v[yi] - v[yj])**2 - (v[ri] + v[rj])**2
                })
        
        return constraints
    
    constraints = make_constraints()
    
    # Bounds: x, y in [0, 1], r in [0, 0.5] (max radius in unit square)
    bounds = []
    for i in range(N):
        bounds.append((0, 1))  # x
        bounds.append((0, 1))  # y
        bounds.append((1e-6, 0.5))  # r
    
    # Multiple random starts
    np.random.seed(42)
    
    max_starts = 50  # We'll try up to this many starts, but check time
    
    for start_idx in range(max_starts):
        # Check deadline
        if time.time() > deadline - 10.0:
            break
        
        # Random initialization
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.random.uniform(0.02, 0.15, N)
        
        v0 = np.zeros(3*N)
        v0[0::3] = x
        v0[1::3] = y
        v0[2::3] = r
        
        try:
            result = minimize(
                objective,
                v0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-10}
            )
            
            if result.success or result.fun < best_value:
                # Check feasibility
                v = result.x
                feasible = True
                
                # Check boundary
                for i in range(N):
                    xi, yi, ri = v[3*i], v[3*i+1], v[3*i+2]
                    if xi < ri - 1e-6 or xi > 1 - ri + 1e-6:
                        feasible = False
                        break
                    if yi < ri - 1e-6 or yi > 1 - ri + 1e-6:
                        feasible = False
                        break
                    if ri < 1e-6:
                        feasible = False
                        break
                
                # Check non-overlap
                if feasible:
                    for i in range(N):
                        for j in range(i+1, N):
                            xi, yi, ri = v[3*i], v[3*i+1], v[3*i+2]
                            xj, yj, rj = v[3*j], v[3*j+1], v[3*j+2]
                            dist_sq = (xi-xj)**2 + (yi-yj)**2
                            if dist_sq < (ri+rj)**2 - 1e-6:
                                feasible = False
                                break
                        if not feasible:
                            break
                
                if feasible:
                    value = np.sum(v[2::3])
                    if value > best_value:
                        best_value = value
                        best_result = v.copy()
                        
        except Exception:
            continue
    
    # If we found a result, format it
    if best_result is not None:
        x = best_result[0::3]
        y = best_result[1::3]
        r = best_result[2::3]
        return np.column_stack([x, y, r])
    else:
        # Fallback: trivial grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

# EVOLVE-BLOCK-END
