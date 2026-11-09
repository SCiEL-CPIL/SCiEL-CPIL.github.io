import numpy as np
from scipy.spatial import cKDTree
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_centers = None
    best_radius = 0.0
    
    # Helper to calculate radius
    def calc_radius(centers):
        # Pairwise distances
        # Using cKDTree is efficient for min dist, but we need the indices for the move
        # For N=100, simple broadcasting is fast enough and gives full matrix
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Wall distances
        # Distance to each wall: 0.5 - |x|, etc.
        # min_i min_k (0.5 - |c_ik|)
        # c_ik is in [-0.5, 0.5]
        # dist to wall is 0.5 - abs(c)
        wall_dists = 0.5 - np.abs(centers)
        min_wall_dist = np.min(wall_dists)
        
        r = min(min_pair_dist / 2.0, min_wall_dist)
        
        # Retrieve indices for the minimum pair if needed
        # For the update step, we just need to know which constraint is active
        # If min_pair_dist/2 < min_wall_dist, the bottleneck is a pair
        # Else it's a wall
        
        return r, min_pair_dist, min_wall_dist

    def optimize(centers, max_iters=1000):
        # Local search: push apart the closest pair or move away from closest wall
        current_radius, min_pair_dist, min_wall_dist = calc_radius(centers)
        
        step_size = 0.01
        # Decay step size? Or keep constant?
        # Let's use a small fixed step and rely on many iterations
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            # Recalculate to find bottleneck
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=-1))
            np.fill_diagonal(dists, np.inf)
            
            # Find min pair
            min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
            min_pair_d = dists[min_pair_idx]
            
            # Find min wall
            wall_dists = 0.5 - np.abs(centers)
            min_wall_idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
            min_wall_d = wall_dists[min_wall_idx]
            
            # Compare effective radii
            # Pair contributes min_pair_d / 2
            # Wall contributes min_wall_d
            if min_pair_d / 2.0 < min_wall_d:
                # Move pair apart
                i, j = min_pair_idx
                # Vector from j to i
                vec = centers[i] - centers[j]
                norm = np.linalg.norm(vec)
                if norm < 1e-10:
                    # Random nudge
                    vec = np.random.normal(size=3)
                    norm = np.linalg.norm(vec)
                vec = vec / norm
                
                # Move i by +vec * step, j by -vec * step
                centers[i] += vec * step_size
                centers[j] -= vec * step_size
            else:
                # Move away from wall
                i, k = min_wall_idx
                # centers[i, k] is the coordinate
                # If centers[i, k] > 0, wall is at 0.5, move in -1 direction
                # If centers[i, k] < 0, wall is at -0.5, move in +1 direction
                if centers[i, k] > 0:
                    centers[i, k] -= step_size
                else:
                    centers[i, k] += step_size
            
            # Clip to bounds
            centers = np.clip(centers, -0.5, 0.5)
            
            # Calculate new radius
            new_radius, _, _ = calc_radius(centers)
            # We don't necessarily need to check if it improved to continue, 
            # as the move is designed to increase the min distance, 
            # but clipping might reduce it.
            
        return centers, calc_radius(centers)[0]

    # Try multiple restarts
    num_restarts = 50
    iters_per_restart = 200
    
    for r_idx in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Initialize
        # Use a grid perturbed by noise, or just random
        # Grid is a good start
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        centers = grid[:N].copy()
        
        # Add small random noise to break symmetry
        centers += np.random.normal(0, 0.01, centers.shape)
        centers = np.clip(centers, -0.5, 0.5)
        
        # Optimize
        centers, rad = optimize(centers, max_iters=iters_per_restart)
        
        if rad > best_radius:
            best_radius = rad
            best_centers = centers.copy()
            
    return best_centers
