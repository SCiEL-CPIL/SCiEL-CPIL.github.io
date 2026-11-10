import numpy as np

N = 50

def construct():
    best_centers = None
    best_radius = 0.0
    
    # Try a few random initializations
    for seed in range(10):
        np.random.seed(seed)
        # Initialize random points in [-0.5, 0.5]
        centers = np.random.uniform(-0.5, 0.5, (N, 3))
        
        # Molecular Dynamics Relaxation
        # Parameters
        steps = 5000
        alpha = 0.01
        # To make it stable, we might need adaptive step size or just small alpha
        
        for step in range(steps):
            # Calculate pairwise distances
            # diff[i, j] = centers[i] - centers[j]
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            
            # Avoid division by zero
            dist[dist == 0] = 1e-6
            
            # Force from pairwise repulsion: F_i = sum_j (r_i - r_j) / dist_ij^2
            # We want to push them apart.
            # factor = 1 / dist^2
            # But we only want strong repulsion if they are close.
            # Using 1/d^2 is standard.
            inv_dist_sq = 1.0 / (dist ** 2)
            # Set diagonal to 0
            np.fill_diagonal(inv_dist_sq, 0.0)
            
            # F_ij = diff_ij * inv_dist_sq_ij
            # F_i = sum_j F_ij
            forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
            
            # Wall forces
            # If x > 0.5 - margin, push left.
            # Let's use a hard wall with a soft penalty.
            # Penalty force: if x > 0.5 - eps, F_x -= 1/(0.5-x)^2
            # This is tricky to vectorize cleanly without conditionals.
            # Simpler: Just clip and let pairwise repulsion do the work?
            # No, spheres will stick to walls.
            
            # Let's add wall repulsion.
            # Left wall x = -0.5. If x < -0.5 + eps, push right.
            # Force = 1 / (x - (-0.5))^2
            # We can compute this for all axes.
            
            # Distance to walls
            # x: dist to -0.5 is x + 0.5, to 0.5 is 0.5 - x
            # y: dist to -0.5 is y + 0.5, to 0.5 is 0.5 - y
            # z: dist to -0.5 is z + 0.5, to 0.5 is 0.5 - z
            
            # We want to repel from walls if distance < some threshold.
            # Let's use a threshold of 0.2 (roughly the expected radius).
            # Actually, just use 1/d^2 for all distances to walls.
            
            d_x_left = centers[:, 0] + 0.5
            d_x_right = 0.5 - centers[:, 0]
            d_y_left = centers[:, 1] + 0.5
            d_y_right = 0.5 - centers[:, 1]
            d_z_left = centers[:, 2] + 0.5
            d_z_right = 0.5 - centers[:, 2]
            
            # Avoid zero division
            eps = 1e-6
            d_x_left = np.maximum(d_x_left, eps)
            d_x_right = np.maximum(d_x_right, eps)
            d_y_left = np.maximum(d_y_left, eps)
            d_y_right = np.maximum(d_y_right, eps)
            d_z_left = np.maximum(d_z_left, eps)
            d_z_right = np.maximum(d_z_right, eps)
            
            # Wall forces
            F_wall = np.zeros_like(centers)
            # Push away from left walls (positive direction)
            F_wall[:, 0] += 1.0 / (d_x_left ** 2)
            F_wall[:, 1] += 1.0 / (d_y_left ** 2)
            F_wall[:, 2] += 1.0 / (d_z_left ** 2)
            
            # Push away from right walls (negative direction)
            F_wall[:, 0] -= 1.0 / (d_x_right ** 2)
            F_wall[:, 1] -= 1.0 / (d_y_right ** 2)
            F_wall[:, 2] -= 1.0 / (d_z_right ** 2)
            
            # Total force
            total_forces = forces + F_wall
            
            # Update positions
            # To prevent explosion, we can normalize forces or use small alpha
            # Let's clip the force magnitude
            force_mags = np.linalg.norm(total_forces, axis=1, keepdims=True)
            force_mags[force_mags == 0] = 1e-6
            # Cap force magnitude
            max_force = 1.0
            total_forces = total_forces / force_mags * np.minimum(force_mags, max_force)
            
            centers += alpha * total_forces
            
            # Clip to box
            centers = np.clip(centers, -0.5, 0.5)
            
        # Calculate radius for this configuration
        # Min pairwise distance / 2
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        # Exclude self
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        
        # Min wall distance
        wall_dists = []
        for k in range(3):
            wall_dists.append(np.min(centers[:, k] + 0.5))
            wall_dists.append(np.min(0.5 - centers[:, k]))
        min_wall_dist = np.min(wall_dists)
        
        radius = min(min_pair_dist / 2.0, min_wall_dist)
        
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
            
    return best_centers
