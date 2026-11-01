import numpy as np
import time

N = 50

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Initialization: Hexagonal lattice
    # We want to pack 50 points. A hexagonal grid is denser than square.
    # Let's generate a hex grid and pick 50 points that are roughly centered.
    # Or simpler: Start with a square grid and perturb, or use a specific hex layout.
    # For N=50, a 7x7 grid is 49 points. Adding 1 is awkward.
    # A 5x10 grid is 50 points.
    # Let's try a hexagonal arrangement.
    
    # Generate hex grid points
    # Row spacing dy = sqrt(3)/2 * dx
    # Let's estimate dx. Area = 1. 50 circles.
    # Approx r ~ 0.07. dx ~ 2*r ~ 0.14.
    # Number of points in width ~ 1/0.14 ~ 7.
    # Number of points in height ~ 1/0.12 ~ 8.
    # 7*8 = 56 points. We need 50.
    
    # Let's create a hex grid of points and select the 50 closest to the center.
    pts = []
    # Generate enough points to cover the square
    # Max coordinate 0.5
    # Let's use a grid of indices
    for i in range(-10, 10):
        for j in range(-10, 10):
            x = i * 0.15 + (j % 2) * 0.075
            y = j * 0.13  # approx sqrt(3)/2 * 0.15
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                pts.append([x, y])
                
    pts = np.array(pts)
    # Sort by distance to center to pick the most central 50
    dists = np.sum(pts**2, axis=1)
    idx = np.argsort(dists)
    centers = pts[idx[:N]].copy()
    
    # 2. Optimization: Force-directed relaxation
    # We want to maximize r.
    # We will iteratively increase a target radius r_target and push circles apart.
    
    r = 0.0  # Current best radius
    r_target = 0.01
    
    # Precompute indices for pairs
    idx_i, idx_j = np.triu_indices(N, k=1)
    
    step_size = 0.01
    cool_rate = 0.999
    
    while time.time() < deadline:
        # Calculate current radius
        # Pair distances
        diffs = centers[idx_i] - centers[idx_j]
        dists = np.linalg.norm(diffs, axis=1)
        
        # Wall distances
        # min(0.5 - x, 0.5 - y, x + 0.5, y + 0.5) for each point
        # x in [-0.5, 0.5]
        wall_dists = np.min(np.stack([
            0.5 - centers[:, 0], 
            centers[:, 0] + 0.5, 
            0.5 - centers[:, 1], 
            centers[:, 1] + 0.5
        ], axis=1), axis=1)
        
        min_pair_dist = np.min(dists)
        min_wall_dist = np.min(wall_dists)
        
        current_r = min(min_pair_dist, min_wall_dist) / 2.0
        
        if current_r > r:
            r = current_r
            r_target = r * 1.005  # Push for slightly larger radius
        else:
            # If we didn't improve, maybe we stuck in local min?
            # For now, keep pushing with current r_target or reduce step size
            r_target = r * 1.001
            step_size *= 0.995
            if step_size < 1e-6:
                # Try a small random kick to escape local minima
                centers += np.random.normal(0, 0.001, centers.shape)
                centers = np.clip(centers, -0.5, 0.5)
                step_size = 0.005
                continue

        # Calculate forces
        forces = np.zeros_like(centers)
        
        # Pair forces
        # If dist < 2 * r_target, push apart
        # Force magnitude proportional to overlap
        overlaps = (2 * r_target - dists)
        mask = overlaps > 0
        if np.any(mask):
            # Normalize direction
            norm_dists = dists[mask]
            # Avoid division by zero
            norm_dists[norm_dists == 0] = 1e-9
            dirs = diffs[mask] / norm_dists[:, np.newaxis]
            # Force vector
            f_mag = overlaps[mask] / (2 * r_target) # Scale force
            f_vecs = f_mag[:, np.newaxis] * dirs
            
            # Scatter add
            for k in range(len(f_vecs)):
                i = idx_i[mask][k]
                j = idx_j[mask][k]
                forces[i] += f_vecs[k]
                forces[j] -= f_vecs[k]
                
        # Wall forces
        # If wall_dist < r_target, push in
        # Determine which wall
        # x < 0: push right (+x)
        # x > 0: push left (-x)
        # y < 0: push up (+y)
        # y > 0: push down (-y)
        
        # Left wall
        mask_l = (0.5 - centers[:, 0]) < r_target
        if np.any(mask_l):
            forces[mask_l, 0] += (r_target - (0.5 - centers[mask_l, 0])) / (2 * r_target)
            
        # Right wall
        mask_r = (centers[:, 0] + 0.5) < r_target
        if np.any(mask_r):
            forces[mask_r, 0] -= (r_target - (centers[mask_r, 0] + 0.5)) / (2 * r_target)
            
        # Bottom wall
        mask_b = (0.5 - centers[:, 1]) < r_target
        if np.any(mask_b):
            forces[mask_b, 1] += (r_target - (0.5 - centers[mask_b, 1])) / (2 * r_target)
            
        # Top wall
        mask_t = (centers[:, 1] + 0.5) < r_target
        if np.any(mask_t):
            forces[mask_t, 1] -= (r_target - (centers[mask_t, 1] + 0.5)) / (2 * r_target)

        # Update positions
        centers += step_size * forces
        centers = np.clip(centers, -0.5, 0.5)
        
        # Decay step size
        step_size *= cool_rate

    return centers

