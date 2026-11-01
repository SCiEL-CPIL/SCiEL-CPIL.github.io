import numpy as np

def construct():
    N = 50
    # Initialize with Fibonacci sphere distribution
    # This provides a nearly uniform distribution on a sphere, breaking symmetry
    # and providing a good starting point for local optimization.
    
    # Radius of the sphere on which we initialize centers
    # Optimal r is ~0.136, so center distance from origin ~ 0.5 - 0.136 = 0.364
    rho_init = 0.364
    
    # Generate Fibonacci sphere points
    golden_ratio = (1 + np.sqrt(5)) / 2
    indices = np.arange(N)
    # Avoid poles for better distribution? Standard Fibonacci lattice covers poles.
    # theta = arccos(1 - 2*(i + 0.5)/N)
    # phi = 2 * pi * i / golden_ratio
    theta = np.arccos(1 - 2 * (indices + 0.5) / N)
    phi = 2 * np.pi * indices / golden_ratio
    
    centers = np.zeros((N, 3))
    centers[:, 0] = rho_init * np.sin(theta) * np.cos(phi)
    centers[:, 1] = rho_init * np.sin(theta) * np.sin(phi)
    centers[:, 2] = rho_init * np.cos(theta)
    
    # Optimization using Repulsive Forces
    # We want to maximize r.
    # At each step, compute forces pushing spheres apart and away from walls.
    # If a sphere is too close to another or a wall, it gets a strong repulsive force.
    
    max_iter = 2000
    # Adaptive step size
    step_size = 0.01
    # Force decay parameter
    force_decay = 0.95
    
    for i in range(max_iter):
        # Compute pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        # Wall distances
        abs_centers = np.abs(centers)
        # Distance to each of the 6 walls
        # w_x_pos = 0.5 - centers[:, 0]
        # w_x_neg = 0.5 + centers[:, 0]
        # ...
        # Min distance to any wall for sphere i
        wall_dists = np.min(np.stack([
            0.5 - centers[:, 0],
            0.5 + centers[:, 0],
            0.5 - centers[:, 1],
            0.5 + centers[:, 1],
            0.5 - centers[:, 2],
            0.5 + centers[:, 2]
        ]), axis=0)
        
        # Current radius estimate
        # Exclude self-distances
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        min_wall_dist = np.min(wall_dists)
        current_r = min(min_pair_dist / 2.0, min_wall_dist)
        
        # Forces
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion
        # Apply force only if distance is less than a threshold (e.g., 2 * current_r * 1.5)
        # To be efficient, we can just apply to all, but decay with distance.
        # F_ij = (1 / d^2) * unit_vector_ij
        for i_idx in range(N):
            for j_idx in range(i_idx + 1, N):
                d = dist[i_idx, j_idx]
                if d < 1e-9:
                    # Random nudge if coincident
                    v = np.random.randn(3)
                else:
                    v = centers[i_idx] - centers[j_idx]
                    v = v / d
                # Force magnitude
                # Stronger if closer
                f_mag = 1.0 / (d * d + 1e-6)
                # Only apply if they are "too close" relative to current best?
                # Or just always apply, but scaled by how much they violate a target?
                # Let's use a target distance of 2 * (current_r + 0.01)
                target_d = 2.0 * (current_r + 0.02)
                if d < target_d:
                    # Scale force by how much we are under target
                    scale = (target_d - d) / target_d
                    forces[i_idx] += scale * v
                    forces[j_idx] -= scale * v
        
        # Wall repulsion
        for i_idx in range(N):
            # Check each wall
            # If close to wall, push away
            # x+ wall: 0.5 - x
            if wall_dists[i_idx] < 0.05:
                # Determine which wall is closest
                dists_to_walls = np.array([
                    0.5 - centers[i_idx, 0],
                    0.5 + centers[i_idx, 0],
                    0.5 - centers[i_idx, 1],
                    0.5 + centers[i_idx, 1],
                    0.5 - centers[i_idx, 2],
                    0.5 + centers[i_idx, 2]
                ])
                min_w_idx = np.argmin(dists_to_walls)
                d_w = dists_to_walls[min_w_idx]
                
                # Force direction
                if min_w_idx == 0: # x+
                    f_dir = np.array([-1, 0, 0])
                elif min_w_idx == 1: # x-
                    f_dir = np.array([1, 0, 0])
                elif min_w_idx == 2: # y+
                    f_dir = np.array([0, -1, 0])
                elif min_w_idx == 3: # y-
                    f_dir = np.array([0, 1, 0])
                elif min_w_idx == 4: # z+
                    f_dir = np.array([0, 0, -1])
                else: # z-
                    f_dir = np.array([0, 0, 1])
                
                # Force magnitude
                target_w = current_r + 0.02
                if d_w < target_w:
                    scale = (target_w - d_w) / target_w
                    forces[i_idx] += scale * f_dir
        
        # Update
        # Normalize forces to prevent huge steps
        force_norm = np.linalg.norm(forces)
        if force_norm > 0:
            forces /= force_norm
            centers += step_size * forces
        
        # Clamp to cube
        centers = np.clip(centers, -0.4999, 0.4999)
        
        # Decay step size
        step_size *= force_decay
        
    return centers
