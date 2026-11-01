import numpy as np
import time
from scipy.spatial.distance import pdist

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initialization
    # Use a hexagonal-like grid by jittering a square grid
    m = int(np.ceil(np.sqrt(N)))
    x = np.linspace(-0.45, 0.45, m)
    y = np.linspace(-0.45, 0.45, m)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel()))[:N]
    
    # Add jitter
    np.random.seed(42)
    points += np.random.normal(0, 0.005, points.shape)
    
    # Initial radius
    r = 0.5 / m * 0.9
    
    best_r = r
    best_points = points.copy()
    
    # Parameters
    base_step = 0.05
    relax_iters = 100
    
    # Pre-allocate arrays if possible? No, N is fixed.
    
    while time.time() < deadline:
        
        # Calculate current radius
        d_pairs = pdist(points)
        min_pair_dist = np.min(d_pairs)
        
        dist_to_boundary = np.minimum(0.5 - np.abs(points[:, 0]), 0.5 - np.abs(points[:, 1]))
        min_bound_dist = np.min(dist_to_boundary)
        
        current_r = min(min_pair_dist, min_bound_dist) / 2.0
        
        if current_r > best_r:
            best_r = current_r
            best_points = points.copy()
            
        if time.time() > deadline - 5:
            break
            
        # Target radius
        # Increase slightly
        target_r = current_r * 1.01 + 1e-6
        
        # Relaxation
        for _ in range(relax_iters):
            if time.time() > deadline - 5:
                break
                
            # Pairwise forces
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            dists_safe = np.where(dists == 0, 1e-9, dists)
            
            overlap = 2 * target_r - dists
            overlap = np.maximum(overlap, 0)
            
            # Force vector
            force_vec = (overlap[:, :, np.newaxis] / dists_safe[:, :, np.newaxis]) * diff
            
            # Zero out diagonal
            mask = ~np.eye(N, dtype=bool)
            force_vec[~mask] = 0
            
            forces = np.sum(force_vec, axis=1)
            
            # Boundary forces
            x = points[:, 0]
            y = points[:, 1]
            
            f_x = np.maximum(0, -0.5 + target_r - x) - np.maximum(0, x - (0.5 - target_r))
            f_y = np.maximum(0, -0.5 + target_r - y) - np.maximum(0, y - (0.5 - target_r))
            
            forces[:, 0] += f_x
            forces[:, 1] += f_y
            
            # Adaptive step size
            norm_forces = np.linalg.norm(forces, axis=1)
            step = base_step / (1 + norm_forces)
            
            points += step * forces
            
            # Clip to square
            points = np.clip(points, -0.5, 0.5)
            
    return best_points
