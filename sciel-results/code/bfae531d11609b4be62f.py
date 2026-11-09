import numpy as np

def construct():
    N = 130
    
    # 1. Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Create a dense grid of test points on the sphere
    # Using a Fibonacci spiral for the grid as well for uniformity
    num_grid = 10000
    j = np.arange(num_grid) + 0.5
    z_g = 1.0 - 2.0 * j / num_grid
    phi_g = np.pi * (1.0 + np.sqrt(5.0)) * j
    r_g = np.sqrt(1.0 - z_g * z_g)
    grid = np.column_stack([r_g * np.cos(phi_g), r_g * np.sin(phi_g), z_g])
    
    # 3. Optimization Loop using Subgradient/Voronoi Centroid approach
    num_iters = 300
    base_lr = 0.1
    
    for it in range(num_iters):
        # Learning rate decay
        lr = base_lr * (0.98 ** it)
        
        # Compute nearest neighbor for each grid point
        # dist[gi, pi] = 1 - dot(grid[gi], pts[pi])
        # Using dot product for speed
        dots = grid @ pts.T  # Shape (num_grid, N)
        # Nearest point is max dot product
        nearest_idx = np.argmax(dots, axis=1)
        
        # Compute distances (angular) for all grid points to their nearest neighbor
        # cos_theta = dots[np.arange(num_grid), nearest_idx]
        cos_theta = dots[np.arange(num_grid), nearest_idx]
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        dists = np.arccos(cos_theta)
        
        # Determine the global covering radius
        R_max = np.max(dists)
        
        # For each configuration point, we want to move it towards the centroid
        # of the grid points assigned to it, weighted by distance to prioritize
        # reducing large gaps.
        
        target_vecs = np.zeros((N, 3))
        weights_sum = np.zeros(N)
        
        # We only care about points that are "far" from the grid points they serve.
        # To focus on the max radius, we weight by (dist / R_max)^power
        # Or simply use all points but weight by distance.
        # Let's use a soft weight: w = exp(dist / R_max * 5)
        # This emphasizes the farthest points in each cell.
        
        weights = np.exp(dists / (R_max + 1e-10) * 5.0)
        
        # Accumulate vectors
        # For each grid point gi, add weight[gi] * grid[gi] to target_vecs[nearest_idx[gi]]
        # We use np.add.at for efficient accumulation
        np.add.at(target_vecs, nearest_idx, grid * weights[:, np.newaxis])
        np.add.at(weights_sum, nearest_idx, weights)
        
        # Update points
        for i in range(N):
            if weights_sum[i] > 1e-10:
                # Centroid direction
                centroid_dir = target_vecs[i] / np.linalg.norm(target_vecs[i])
                # Move point towards centroid
                # We need to be careful: moving towards the centroid of the *entire* cell
                # might not reduce the *max* distance if the cell is large.
                # However, it is a standard heuristic.
                pts[i] = pts[i] + lr * centroid_dir
                pts[i] = pts[i] / np.linalg.norm(pts[i])
            else:
                # Should not happen, but if so, jitter
                pts[i] += 1e-5 * np.random.randn(3)
                pts[i] = pts[i] / np.linalg.norm(pts[i])
                
        # Global normalization
        norms = np.linalg.norm(pts, axis=1)
        pts = pts / norms[:, np.newaxis]
        
    return pts
