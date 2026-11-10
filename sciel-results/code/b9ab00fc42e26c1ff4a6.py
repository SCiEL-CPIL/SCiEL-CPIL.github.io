import numpy as np
from scipy.optimize import minimize

N = 50

def construct():
    best_centers = None
    best_radius = 0.0
    
    # Objective function to minimize
    # We minimize the potential energy, which is a proxy for maximizing the minimum distance.
    # E = sum_{i<j} 1/d_ij^2 + sum_{i,k} 1/w_ik^2
    # where w_ik is the distance from center i to the wall in direction k.
    
    def potential_energy(centers_flat):
        centers = centers_flat.reshape(N, 3)
        
        # Pairwise repulsion
        # diff[i, j] = centers[i] - centers[j]
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        
        # Avoid division by zero
        # We only care about i < j
        # dist_sq[i, i] is 0, we ignore it.
        # For numerical stability, set diagonal to a large number or mask it.
        mask = ~np.eye(N, dtype=bool)
        dist_sq_masked = np.where(mask, dist_sq, 1e10)
        
        # Sum of inverse squared distances
        # We sum over all i,j but divide by 2 to count each pair once? 
        # Actually, sum_{i<j} 1/d^2 is half of sum_{i!=j} 1/d^2.
        # Since we want to minimize, the factor doesn't matter for the location of the minimum, just the scale.
        # However, large values might cause overflow.
        # Let's use 1/d^2.
        pair_energy = np.sum(1.0 / dist_sq_masked)
        
        # Wall repulsion
        # Distances to walls:
        # x: x + 0.5, 0.5 - x
        # y: y + 0.5, 0.5 - y
        # z: z + 0.5, 0.5 - z
        
        # We want to penalize being close to walls.
        # d_wall_min for each coordinate.
        # For x: min(x+0.5, 0.5-x)
        # But we want a smooth potential.
        # Let's use sum of inverse squared distances to ALL 6 walls.
        
        dx_left = centers[:, 0] + 0.5
        dx_right = 0.5 - centers[:, 0]
        dy_left = centers[:, 1] + 0.5
        dy_right = 0.5 - centers[:, 1]
        dz_left = centers[:, 2] + 0.5
        dz_right = 0.5 - centers[:, 2]
        
        # Clip to avoid division by zero if exactly on wall (shouldn't happen with bounds, but good for safety)
        eps = 1e-8
        dx_left = np.maximum(dx_left, eps)
        dx_right = np.maximum(dx_right, eps)
        dy_left = np.maximum(dy_left, eps)
        dy_right = np.maximum(dy_right, eps)
        dz_left = np.maximum(dz_left, eps)
        dz_right = np.maximum(dz_right, eps)
        
        wall_energy = np.sum(1.0 / dx_left**2 + 1.0 / dx_right**2 + 
                             1.0 / dy_left**2 + 1.0 / dy_right**2 + 
                             1.0 / dz_left**2 + 1.0 / dz_right**2)
        
        return pair_energy + wall_energy

    def calculate_radius(centers):
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        
        wall_dists = []
        for k in range(3):
            wall_dists.append(np.min(centers[:, k] + 0.5))
            wall_dists.append(np.min(0.5 - centers[:, k]))
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Try multiple initializations
    for seed in range(20):
        np.random.seed(seed)
        # Random initialization
        centers_init = np.random.uniform(-0.4, 0.4, (N, 3))
        centers_flat = centers_init.flatten()
        
        # Optimize
        try:
            res = minimize(potential_energy, centers_flat, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            centers_opt = res.x.reshape(N, 3)
            
            # Ensure centers are strictly inside (clip just in case of numerical drift)
            centers_opt = np.clip(centers_opt, -0.5 + 1e-9, 0.5 - 1e-9)
            
            radius = calculate_radius(centers_opt)
            
            if radius > best_radius:
                best_radius = radius
                best_centers = centers_opt.copy()
                
        except Exception as e:
            pass

    return best_centers
