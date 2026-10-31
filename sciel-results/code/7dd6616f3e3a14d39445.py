"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random

def construct():
    """
    Pack 26 circles with disjoint interiors inside [0,1]^2 and maximize the sum of their radii.
    Uses a combination of initial heuristics and gradient-based optimization.
    """
    # Initial positions: random or grid-based
    # We'll use a few different initializations and pick the best one after optimization.
    
    def pack_circles(init_positions):
        n = len(init_positions)
        x = np.array(init_positions, dtype=float)
        
        # Parameters: 2n coordinates + n radii
        # We'll optimize positions and radii simultaneously
        # Objective: maximize sum of radii
        # Constraints: 
        #   1. Circles inside [0,1]^2: x_i - r_i >= 0, x_i + r_i <= 1, y_i - r_i >= 0, y_i + r_i <= 1
        #   2. Disjoint interiors: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2 for i != j
        
        def objective(params):
            # params: [x_0, y_0, r_0, x_1, y_1, r_1, ..., x_{n-1}, y_{n-1}, r_{n-1}]
            radii = params[2::3]
            return -np.sum(radii)
        
        def constraints(params):
            cons = []
            for i in range(n):
                xi = params[3*i]
                yi = params[3*i+1]
                ri = params[3*i+2]
                # Inside unit square
                cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2]})  # x_i - r_i >= 0
                cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[3*i] - p[3*i+2]})  # 1 - x_i - r_i >= 0
                cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2]})  # y_i - r_i >= 0
                cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[3*i+1] - p[3*i+2]})  # 1 - y_i - r_i >= 0
                # Radii positive
                cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+2]})
            
            # Disjointness constraints
            for i in range(n):
                for j in range(i+1, n):
                    xi = params[3*i]
                    yi = params[3*i+1]
                    ri = params[3*i+2]
                    xj = params[3*j]
                    yj = params[3*j+1]
                    rj = params[3*j+2]
                    cons.append({'type': 'ineq', 'fun': lambda p, i=i, j=j: (p[3*i]-p[3*j])**2 + (p[3*i+1]-p[3*j+1])**2 - (p[3*i+2]+p[3*j+2])**2})
            
            return cons
        
        # Initial parameters
        params0 = []
        for i in range(n):
            params0.extend([x[i, 0], x[i, 1], 0.05])  # Small initial radius
        
        # Use SLSQP or trust-constr
        try:
            result = minimize(objective, params0, method='SLSQP', constraints=constraints, 
                             bounds=[(0, 1)] * (2*n) + [(0, 0.5)] * n,
                             options={'maxiter': 500, 'ftol': 1e-8})
            if result.success:
                best_params = result.x
            else:
                best_params = params0
        except:
            best_params = params0
        
        # Extract positions and radii
        result_array = np.zeros((n, 3))
        for i in range(n):
            result_array[i, 0] = best_params[3*i]
            result_array[i, 1] = best_params[3*i+1]
            result_array[i, 2] = max(0, best_params[3*i+2])
        
        return result_array
    
    # Try multiple initializations
    best_result = None
    best_score = -np.inf
    
    # Initialization 1: Grid
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            cells.append([((i + 0.5) / side, (j + 0.5) / side)])
    init1 = np.array(cells[:N])
    
    # Initialization 2: Random
    np.random.seed(42)
    init2 = np.random.uniform(0.1, 0.9, (N, 2))
    
    # Initialization 3: Hexagonal-like packing
    init3 = np.zeros((N, 2))
    rows = 5
    cols = 6
    dx = 1.0 / cols
    dy = 1.0 / rows
    for i in range(N):
        row = i // cols
        col = i % cols
        offset = dx * 0.5 if row % 2 == 1 else 0
        init3[i, 0] = (col + 0.5) * dx + offset
        init3[i, 1] = (row + 0.5) * dy
    
    # Try each initialization
    for init in [init1, init2, init3]:
        result = pack_circles(init)
        score = np.sum(result[:, 2])
        if score > best_score:
            best_score = score
            best_result = result
    
    # Validate and clip to ensure validity
    if best_result is not None:
        # Ensure circles are inside the unit square
        for i in range(N):
            x, y, r = best_result[i]
            # Clip radius to fit in square
            max_r = min(x, 1-x, y, 1-y)
            if r > max_r:
                r = max(0, max_r)
            best_result[i] = [x, y, r]
        
        # Remove any overlapping circles by reducing radii
        for iteration in range(100):
            modified = False
            for i in range(N):
                for j in range(i+1, N):
                    xi, yi, ri = best_result[i]
                    xj, yj, rj = best_result[j]
                    dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                    if dist < ri + rj:
                        # Reduce radii to avoid overlap
                        overlap = ri + rj - dist
                        ri -= overlap/2
                        rj -= overlap/2
                        ri = max(0, ri)
                        rj = max(0, rj)
                        best_result[i, 2] = ri
                        best_result[j, 2] = rj
                        modified = True
            if not modified:
                break
    
    return best_result if best_result is not None else np.zeros((N, 3))
# EVOLVE-BLOCK-END
