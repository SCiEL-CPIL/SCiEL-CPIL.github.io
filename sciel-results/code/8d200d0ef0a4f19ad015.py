import numpy as np
from scipy.optimize import minimize

N = 50

def construct():
    best_centers = None
    best_radius = 0.0
    
    # Try a few random initializations with different strategies
    for seed in range(20):
        np.random.seed(seed)
        
        # Different initialization strategies
        if seed < 5:
            # Random uniform
            centers = np.random.uniform(-0.45, 0.45, (N, 3))
        elif seed < 10:
            # Grid-based with jitter
            n_per_dim = 4
            grid_points = []
            for i in range(n_per_dim):
                for j in range(n_per_dim):
                    for k in range(n_per_dim):
                        grid_points.append([
                            -0.4 + i * 0.8 / (n_per_dim - 1),
                            -0.4 + j * 0.8 / (n_per_dim - 1),
                            -0.4 + k * 0.8 / (n_per_dim - 1)
                        ])
            grid_points = np.array(grid_points[:N])
            centers = grid_points + np.random.uniform(-0.05, 0.05, (N, 3))
            centers = np.clip(centers, -0.45, 0.45)
        else:
            # Random but with some structure
            centers = np.random.uniform(-0.45, 0.45, (N, 3))
            # Sort by one coordinate to create some ordering
            idx = np.argsort(centers[:, 0])
            centers = centers[idx]
        
        # Define objective function: minimize negative radius
        # radius = min(min_pair_dist/2, min_wall_dist)
        # We want to maximize this, so minimize -radius
        
        def objective(x):
            c = x.reshape(N, 3)
            # Pairwise distances
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dist, np.inf)
            min_pair_dist = np.min(dist)
            
            # Wall distances
            min_wall_dist = np.min([
                np.min(c[:, k] + 0.5) for k in range(3)
            ])
            min_wall_dist = min(min_wall_dist, np.min([
                np.min(0.5 - c[:, k]) for k in range(3)
            ]))
            
            radius = min(min_pair_dist / 2.0, min_wall_dist)
            return -radius
        
        # Constraints: keep points inside cube
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        # Use L-BFGS-B for optimization
        result = minimize(objective, centers.flatten(), method='L-BFGS-B', 
                         bounds=bounds, options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        
        optimized_centers = result.x.reshape(N, 3)
        
        # Calculate radius
        diff = optimized_centers[:, np.newaxis, :] - optimized_centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        
        wall_dists = []
        for k in range(3):
            wall_dists.append(np.min(optimized_centers[:, k] + 0.5))
            wall_dists.append(np.min(0.5 - optimized_centers[:, k]))
        min_wall_dist = np.min(wall_dists)
        
        radius = min(min_pair_dist / 2.0, min_wall_dist)
        
        if radius > best_radius:
            best_radius = radius
            best_centers = optimized_centers.copy()
    
    return best_centers
