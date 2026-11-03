import numpy as np
import time
from scipy.optimize import minimize

N = 200

# EVOLVE-BLOCK-START
def construct():
    """
    Uses a multi-stage optimization approach:
    1. Start with a good grid initialization
    2. Use scipy's L-BFGS-B to maximize the minimum radius
    3. Use local refinement with Nelder-Mead on the bottleneck
    """
    
    def objective(points_flat):
        """
        Negative of the radius (to be minimized).
        r = min(min_pair_dist / 2, min_wall_dist)
        """
        points = points_flat.reshape(-1, 2)
        
        # Wall distances
        wall_dists = np.minimum.reduce([
            0.5 - points[:, 0],
            0.5 + points[:, 0],
            0.5 - points[:, 1],
            0.5 + points[:, 1]
        ])
        min_wall = np.min(wall_dists)
        
        # Pair distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        r = min(min_wall, min_pair / 2.0)
        return -r
    
    def evaluate(points):
        """Returns the radius for a configuration."""
        return -objective(points.flatten())
    
    # Initialization: grid with slight jitter
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    points = grid[:N].copy()
    points += np.random.normal(0, 0.005, size=points.shape)
    points = np.clip(points, -0.49, 0.49)
    
    best_points = points.copy()
    best_score = evaluate(best_points)
    
    # Time limit
    start_time = time.time()
    time_limit = 115  # Leave 5s buffer
    
    # Multi-start L-BFGS-B optimization
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Create perturbed initial point
        if restart == 0:
            x0 = best_points.flatten()
        else:
            x0 = best_points.flatten() + np.random.normal(0, 0.02, size=N * 2)
            x0 = np.clip(x0.reshape(-1, 2), -0.49, 0.49).flatten()
        
        # Bounds: each coordinate in [-0.5, 0.5]
        bounds = [(-0.5, 0.5)] * (N * 2)
        
        # Run L-BFGS-B
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            new_points = result.x.reshape(-1, 2)
            new_score = evaluate(new_points)
            
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
        except Exception:
            pass
    
    # Local refinement: focus on bottleneck pairs
    for _ in range(3):
        if time.time() - start_time > time_limit:
            break
        
        # Find bottleneck
        points = best_points
        wall_dists = np.minimum.reduce([
            0.5 - points[:, 0],
            0.5 + points[:, 0],
            0.5 - points[:, 1],
            0.5 + points[:, 1]
        ])
        min_wall_val = np.min(wall_dists)
        
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_val = np.min(dists)
        
        # Try small perturbations on all points
        for trial in range(100):
            if time.time() - start_time > time_limit:
                break
            
            # Random perturbation
            scale = 0.005 * (1 - trial / 100)
            new_points = best_points + np.random.normal(0, scale, size=best_points.shape)
            new_points = np.clip(new_points, -0.5, 0.5)
            
            new_score = evaluate(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
    
    return best_points
# EVOLVE-BLOCK-END
