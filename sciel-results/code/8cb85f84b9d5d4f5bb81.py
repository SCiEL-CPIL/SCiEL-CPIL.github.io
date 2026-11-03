import numpy as np
import scipy.optimize as so
import time

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    # 1. Generate Hexagonal Grid Initial Guess
    # Hexagonal packing density is pi / (2 * sqrt(3)) ~ 0.9069
    # Area per circle ~ 1/200
    # Spacing s such that s^2 * (sqrt(3)/2) ~ 1/200 => s ~ sqrt(2/(200*sqrt(3))) ~ 0.075
    
    best_centers = None
    best_score = -np.inf

    # Try a few different spacings to ensure we get enough points
    spacings = [0.075, 0.076, 0.077, 0.078, 0.08]
    
    for s in spacings:
        if time.time() > deadline:
            break
            
        # Generate hex grid points
        # Hex grid coordinates:
        # x = i * s + (j % 2) * s / 2
        # y = j * s * sqrt(3) / 2
        
        # Estimate range needed
        # Side length 1. Range approx 1/s
        max_idx = int(1.0 / s) + 2
        
        points = []
        for j in range(-max_idx, max_idx + 1):
            for i in range(-max_idx, max_idx + 1):
                x = i * s + (j % 2) * s * 0.5
                y = j * s * (np.sqrt(3) / 2)
                points.append((x, y))
                
        points = np.array(points)
        
        # Filter points inside the square [-0.5, 0.5]
        # We want to be strictly inside to allow for optimization movement
        # But for initialization, let's take points close to the boundary as well
        mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
        points = points[mask]
        
        if len(points) < N:
            continue
            
        # If we have more than N points, select the best N
        # Heuristic: Select points closest to the center to maintain a compact shape
        # This helps in avoiding empty corners which might lower the min distance
        dists_to_center = np.linalg.norm(points, axis=1)
        sorted_indices = np.argsort(dists_to_center)
        points = points[sorted_indices[:N]]
        
        # 2. Optimize
        # Objective: Maximize min(pair_dist/2, wall_dist)
        # Equivalent to Minimize -min(...)
        
        def objective(x_flat):
            centers = x_flat.reshape(N, 2)
            
            # Pairwise distances
            # Using broadcasting
            # (N, 1, 2) - (1, N, 2) -> (N, N, 2)
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            
            # Exclude self-distances
            np.fill_diagonal(dists, np.inf)
            
            # Min pair distance
            min_pair_dist = np.min(dists)
            
            # Wall distances
            # Distance to x = -0.5 is centers[:, 0] + 0.5
            # Distance to x = 0.5 is 0.5 - centers[:, 0]
            # Distance to y = -0.5 is centers[:, 1] + 0.5
            # Distance to y = 0.5 is 0.5 - centers[:, 1]
            
            wall_dists = np.minimum(np.minimum(centers[:, 0] + 0.5, 0.5 - centers[:, 0]),
                                    np.minimum(centers[:, 1] + 0.5, 0.5 - centers[:, 1]))
            
            min_wall_dist = np.min(wall_dists)
            
            # The radius is half the min pair dist, and also limited by min wall dist
            # r = min(min_pair_dist / 2, min_wall_dist)
            r = min(min_pair_dist / 2.0, min_wall_dist)
            
            return -r
        
        # Use a smooth approximation to help the optimizer?
        # L-BFGS-B handles non-smooth okay, but let's try standard first.
        # If it stalls, we can switch to soft-min.
        
        x0 = points.flatten()
        
        # Run optimization
        # Options for L-BFGS-B
        options = {
            'maxiter': 2000,
            'ftol': 1e-12,
            'gtol': 1e-9,
            'maxfun': 10000
        }
        
        try:
            res = so.minimize(objective, x0, method='L-BFGS-B', options=options)
            x_opt = res.x
            score = -res.fun
            
            if score > best_score:
                best_score = score
                best_centers = x_opt.reshape(N, 2)
                
        except Exception as e:
            pass
        
        # Check time
        if time.time() > deadline:
            break

    # If no valid center found (should not happen with grid), fallback to grid
    if best_centers is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    # Final Polish: A few more iterations with tighter tolerance on the best result
    if time.time() < deadline - 5 and best_centers is not None:
        def objective_polish(x_flat):
            centers = x_flat.reshape(N, 2)
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            min_pair_dist = np.min(dists)
            
            wall_dists = np.minimum(np.minimum(centers[:, 0] + 0.5, 0.5 - centers[:, 0]),
                                    np.minimum(centers[:, 1] + 0.5, 0.5 - centers[:, 1]))
            min_wall_dist = np.min(wall_dists)
            
            r = min(min_pair_dist / 2.0, min_wall_dist)
            return -r
            
        try:
            res = so.minimize(objective_polish, best_centers.flatten(), method='L-BFGS-B', 
                              options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
            if -res.fun > best_score:
                best_centers = res.x.reshape(N, 2)
        except Exception:
            pass

    return best_centers
