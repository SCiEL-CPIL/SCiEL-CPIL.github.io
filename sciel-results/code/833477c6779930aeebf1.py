"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit
    
    best_score = -np.inf
    best_params = None
    
    # Vectorized Objective function
    def objective(z):
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        
        # Boundary penalty
        # Shape (N, 4)
        db = np.column_stack((
            r - x, 
            r - (1 - x), 
            r - y, 
            r - (1 - y)
        ))
        pen_bound = np.sum(np.maximum(0, db)**2)
        
        # Overlap penalty
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        
        r_sum = r[:, None] + r[None, :]
        
        overlap = r_sum - dist
        np.fill_diagonal(overlap, 0)
        
        pen_overlap = np.sum(np.maximum(0, overlap)**2)
        
        # Maximize sum of radii -> Minimize negative sum
        # High penalty weight to enforce constraints
        return -np.sum(r) - 1000.0 * (pen_bound + pen_overlap)

    # Bounds for L-BFGS-B
    # x, y in [0, 1], r in [0, 0.5]
    bounds = [(0, 1)] * N + [(0, 1)] * N + [(0, 0.5)] * N

    # Initializations
    inits = []
    
    # 1. Grid 6x5
    side_x, side_y = 6, 5
    cells = []
    for i in range(side_x):
        for j in range(side_y):
            cells.append([(i + 0.5) / side_x, (j + 0.5) / side_y, 0.5 / side_x])
    init_grid = np.array(cells[:N]).flatten()
    inits.append(init_grid)
    
    # 2. Random
    np.random.seed(42)
    init_rand = np.random.rand(N * 3)
    init_rand[2::3] *= 0.1  # Small initial radii
    inits.append(init_rand)
    
    # 3. Perturbed Grid
    init_perturb = init_grid + np.random.rand(N * 3) * 0.05
    inits.append(init_perturb)
    
    # 4. Another Grid 5x6 (transposed)
    side_x, side_y = 5, 6
    cells = []
    for i in range(side_x):
        for j in range(side_y):
            cells.append([(i + 0.5) / side_x, (j + 0.5) / side_y, 0.5 / side_y])
    init_grid2 = np.array(cells[:N]).flatten()
    inits.append(init_grid2)

    for idx, init in enumerate(inits):
        if time.time() > deadline:
            break
            
        try:
            res = minimize(objective, init, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-7})
            
            # Check validity and repair
            x = res.x[0::3].copy()
            y = res.x[1::3].copy()
            r = res.x[2::3].copy()
            
            # Repair boundaries: clip radii
            r = np.minimum(r, np.minimum(x, 1 - x))
            r = np.minimum(r, np.minimum(y, 1 - y))
            r = np.maximum(r, 0)
            
            # Repair overlaps: iteratively reduce radii
            for _ in range(200):
                # Calculate overlaps
                dx = x[:, None] - x[None, :]
                dy = y[:, None] - y[None, :]
                dist = np.sqrt(dx**2 + dy**2)
                r_sum = r[:, None] + r[None, :]
                overlap = r_sum - dist
                np.fill_diagonal(overlap, 0)
                
                max_overlap = np.max(overlap)
                if max_overlap <= 1e-9:
                    break
                
                # Find pair with max overlap
                i, j = np.unravel_index(np.argmax(overlap), overlap.shape)
                if i > j: i, j = j, i
                
                # Reduce radii
                # Reduce both to resolve overlap
                # r_i + r_j > d_ij
                # Scale down
                current_sum = r[i] + r[j]
                target_sum = dist[i, j]
                if current_sum > 0:
                    scale = target_sum / current_sum
                    # Apply scale, maybe with a factor < 1 to be safe
                    r[i] *= scale * 0.99
                    r[j] *= scale * 0.99
                else:
                    r[i] = 0
                    r[j] = 0
            
            score = np.sum(r)
            if score > best_score:
                best_score = score
                # Store valid params
                valid_params = np.zeros(N * 3)
                valid_params[0::3] = x
                valid_params[1::3] = y
                valid_params[2::3] = r
                best_params = valid_params
                
        except Exception:
            pass

    if best_params is None:
        # Fallback: Grid 6x5 with small radii
        side_x, side_y = 6, 5
        cells = []
        for i in range(side_x):
            for j in range(side_y):
                cells.append([(i + 0.5) / side_x, (j + 0.5) / side_y, 0.4 / side_x])
        best_params = np.array(cells[:N]).flatten()
        
    # Format output
    x = best_params[0::3]
    y = best_params[1::3]
    r = best_params[2::3]
    
    result = np.column_stack((x, y, r))
    return result
# EVOLVE-BLOCK-END
