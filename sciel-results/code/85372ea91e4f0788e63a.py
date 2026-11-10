import numpy as np
from scipy.optimize import minimize

N = 50

def construct():
    # Known best radius
    r_best = 0.1359545126417
    
    # We will try to optimize for a target radius slightly lower than best
    # to ensure we find a valid configuration, then maybe push higher.
    # But usually, optimizing for the best known value works well to find a local optimum near it.
    
    best_score = 0
    best_pos = None
    
    # Multiple restarts to escape local minima
    num_restarts = 5
    
    # Define the smooth objective function
    def objective(x, r_target):
        c = x.reshape(N, 3)
        
        # Pairwise distances
        # Using broadcasting
        d = np.linalg.norm(c[:, np.newaxis] - c[np.newaxis, :], axis=2)
        # Set diagonal to inf to ignore self-distance
        np.fill_diagonal(d, np.inf)
        
        # Wall clearances
        w = 0.5 - np.abs(c)
        
        # Smooth violation terms
        # We want d >= 2*r_target and w >= r_target
        # viol_d = max(0, 2*r_target - d)
        # Use softplus for smoothness: log(1 + exp(x))
        # To avoid overflow, we can clip x
        x_d = 2 * r_target - d
        x_w = r_target - w
        
        # Clip to prevent numerical issues in exp
        x_d = np.clip(x_d, -50, 50)
        x_w = np.clip(x_w, -50, 50)
        
        viol_d = np.log1p(np.exp(x_d))
        viol_w = np.log1p(np.exp(x_w))
        
        return np.sum(viol_d**2) + np.sum(viol_w**2)
    
    # Bounds for each coordinate: [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    for restart in range(num_restarts):
        # Initialization: Jittered Grid
        # Create a 4x4x4 grid of cells
        m = 4
        cell_size = 1.0 / m
        
        # Generate all possible cell centers
        # We need to pick N cells out of m^3
        # To ensure good spread, we can pick a subset of the grid indices
        # A simple way: generate a grid of m^3 points, shuffle, take N, and jitter
        
        grid_pts = []
        for i in range(m):
            for j in range(m):
                for k in range(m):
                    # Center of the cell
                    x = -0.5 + (i + 0.5) * cell_size
                    y = -0.5 + (j + 0.5) * cell_size
                    z = -0.5 + (k + 0.5) * cell_size
                    grid_pts.append([x, y, z])
        
        grid_pts = np.array(grid_pts)
        
        # Shuffle and pick N
        np.random.shuffle(grid_pts)
        init_pts = grid_pts[:N]
        
        # Add jitter to break symmetry
        # Jitter should be small enough to stay within cell but large enough to escape grid
        jitter = np.random.uniform(-0.1, 0.1, init_pts.shape)
        init_pts += jitter
        
        # Clip to bounds
        init_pts = np.clip(init_pts, -0.5 + 1e-6, 0.5 - 1e-6)
        
        x0 = init_pts.flatten()
        
        # Run optimization
        # Try a target radius close to the best known
        # If we target too high, it might fail to converge to a valid state, but still improve the packing.
        # Let's target 0.135
        r_target = 0.135
        
        res = minimize(objective, x0, args=(r_target,), method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        
        # Evaluate the resulting configuration
        c = res.x.reshape(N, 3)
        
        # Calculate actual score
        d = np.linalg.norm(c[:, np.newaxis] - c[np.newaxis, :], axis=2)
        np.fill_diagonal(d, np.inf)
        min_d = np.min(d)
        
        w = 0.5 - np.abs(c)
        min_w = np.min(w)
        
        score = min(min_d / 2.0, min_w)
        
        if score > best_score:
            best_score = score
            best_pos = c.copy()
            
            # If we found a valid configuration for the target radius,
            # we might try to push the radius higher in subsequent restarts or a polish step
            if score >= r_target - 1e-6:
                # Try a polish step with a higher target
                r_target_high = r_best * 0.999
                res2 = minimize(objective, x0, args=(r_target_high,), method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
                c2 = res2.x.reshape(N, 3)
                d2 = np.linalg.norm(c2[:, np.newaxis] - c2[np.newaxis, :], axis=2)
                np.fill_diagonal(d2, np.inf)
                min_d2 = np.min(d2)
                w2 = 0.5 - np.abs(c2)
                min_w2 = np.min(w2)
                score2 = min(min_d2 / 2.0, min_w2)
                
                if score2 > best_score:
                    best_score = score2
                    best_pos = c2.copy()

    # Final check: ensure best_pos is valid
    if best_pos is None:
        # Fallback to seed
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_pos = grid[:N]
        
    return best_pos
