import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 26

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin
    
    best_score = -np.inf
    best_solution = None
    
    # Objective function to minimize
    # z is flattened [x1, y1, r1, x2, y2, r2, ...]
    def objective(z):
        # Unflatten
        z = z.reshape(N, 3)
        x = z[:, 0]
        y = z[:, 1]
        r = z[:, 2]
        
        # Sum of radii (we minimize negative sum)
        sum_r = np.sum(r)
        
        # Overlap penalty
        # Calculate pairwise distances
        dist = cdist(np.column_stack((x, y)), np.column_stack((x, y)))
        # We need upper triangle
        iu = np.triu_indices(N, k=1)
        d_ij = dist[iu]
        r_i = r[iu[0]]
        r_j = r[iu[1]]
        overlaps = np.maximum(0, r_i + r_j - d_ij)
        penalty_overlap = np.sum(overlaps**2)
        
        # Boundary penalty
        # r_i <= x_i, r_i <= 1-x_i, etc.
        pen_b = 0.0
        pen_b += np.sum(np.maximum(0, r - x)**2)
        pen_b += np.sum(np.maximum(0, r - (1 - x))**2)
        pen_b += np.sum(np.maximum(0, r - y)**2)
        pen_b += np.sum(np.maximum(0, r - (1 - y))**2)
        
        # Total penalty
        penalty = penalty_overlap + pen_b
        
        return -sum_r + 100.0 * penalty

    # Gradient (optional, L-BFGS-B can use numerical, but analytical is faster)
    # For simplicity and robustness in this rewrite, we might skip analytical gradient 
    # unless we are sure. L-BFGS-B with numerical gradient for 78 variables is okay for a few iterations.
    # However, to be safe on time, we'll rely on numerical gradient but limit iterations.
    
    def run_optimization(initial_z):
        # Bounds for each variable
        # x in [0,1], y in [0,1], r in [1e-6, 0.5]
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0)) # x
            bounds.append((0.0, 1.0)) # y
            bounds.append((1e-6, 0.5)) # r
            
        res = minimize(objective, initial_z, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        return res

    # Generate initial guesses
    # 1. Perturbed grid
    side = int(np.ceil(np.sqrt(N)))
    # Adjust side to fit N better? 5x5=25, so one extra.
    # Let's just use a 5x5 grid and drop one, or use 5x6 and drop 4.
    # A 5x5 grid has 25 spots. We need 26.
    # Let's create a 5x6 grid, take first 26.
    coords = []
    for i in range(5):
        for j in range(6):
            coords.append((i, j))
    # This creates 30 spots. We take 26.
    # Better: Use a hexagonal-like init or just random.
    
    inits = []
    
    # Init 1: Grid 5x5 + 1 extra
    # Center of 5x5 grid
    for i in range(5):
        for j in range(5):
            inits.append([ (i+0.5)/5.0, (j+0.5)/5.0, 0.1 ])
    # Add one more in the center or corner? Let's put it in the middle of an edge or just random.
    inits.append([0.5, 0.5, 0.1]) # This might overlap with center grid point (2,2) which is 0.5,0.5.
    # Let's shift the grid slightly or use random init for the extra one.
    inits[-1] = [0.1, 0.1, 0.1] # Corner
    
    # Init 2: Random
    np.random.seed(42)
    rand_x = np.random.rand(N)
    rand_y = np.random.rand(N)
    rand_r = np.full(N, 0.05)
    rand_init = np.column_stack((rand_x, rand_y, rand_r)).flatten()
    inits.append(rand_init)
    
    # Init 3: Hexagonal packing approximation
    # For N=26, a hexagonal packing is dense.
    # Let's just use a few random restarts with different seeds.
    for seed in [1, 2, 3, 4, 5]:
        np.random.seed(seed)
        rx = np.random.rand(N)
        ry = np.random.rand(N)
        rr = np.full(N, 0.08) # Slightly larger start
        inits.append(np.column_stack((rx, ry, rr)).flatten())

    # Flatten inits
    for init in inits:
        if time.time() - start_time > timeout:
            break
            
        z0 = np.array(init).flatten()
        try:
            res = run_optimization(z0)
            if res.success:
                z_opt = res.x.reshape(N, 3)
                # Validate
                # Check if penalty is low enough
                val = objective(z_opt)
                # The actual score is sum of radii.
                sum_r = np.sum(z_opt[:, 2])
                # We need to ensure validity. 
                # If penalty is 0, it's valid.
                # Let's extract penalty part
                z = z_opt
                x, y, r = z[:,0], z[:,1], z[:,2]
                dist = cdist(np.column_stack((x, y)), np.column_stack((x, y)))
                iu = np.triu_indices(N, k=1)
                d_ij = dist[iu]
                r_i, r_j = r[iu[0]], r[iu[1]]
                max_overlap = np.max(np.maximum(0, r_i + r_j - d_ij))
                
                pen_b = 0.0
                pen_b += np.max(np.maximum(0, r - x))
                pen_b += np.max(np.maximum(0, r - (1 - x)))
                pen_b += np.max(np.maximum(0, r - y))
                pen_b += np.max(np.maximum(0, r - (1 - y)))
                
                if max_overlap < 1e-5 and pen_b < 1e-5:
                    if sum_r > best_score:
                        best_score = sum_r
                        best_solution = z_opt
        except Exception:
            pass

    if best_solution is None:
        # Fallback to trivial seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_solution
