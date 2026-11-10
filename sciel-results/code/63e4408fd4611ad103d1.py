"""Improved program for packing 200 equal circles in a unit square."""
import numpy as np
import time
from scipy.spatial.distance import pdist

N = 200

def calculate_radius(centers):
    """Calculate the radius of the packing."""
    if N < 2:
        min_dist = np.inf
    else:
        min_dist = np.min(pdist(centers, metric='euclidean'))
    
    dx = np.abs(centers[:, 0])
    dy = np.abs(centers[:, 1])
    dist_boundary = np.min(np.column_stack([0.5 - dx, 0.5 - dy]), axis=1)
    min_boundary_dist = np.min(dist_boundary)
    
    radius = min(min_dist / 2.0, min_boundary_dist)
    return radius

def construct():
    """
    Construct centers of 200 equal non-overlapping circles in [-0.5, 0.5]^2
    maximizing the common radius.
    """
    best_centers = None
    best_radius = -1.0
    start_time = time.time()
    time_limit = 110.0
    
    num_restarts = 25
    steps_per_restart = 4000
    
    golden_ratio = (1 + np.sqrt(5)) / 2
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Fibonacci lattice initialization
        i = np.arange(N)
        r = np.sqrt((i + 0.5) / N)
        theta = 2 * np.pi * (i + 0.5) / golden_ratio
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Scale to fit within square
        max_coord = np.max(np.abs(np.column_stack([x, y])))
        if max_coord > 0:
            scale = 0.48 / max_coord
            x *= scale
            y *= scale
        
        centers = np.column_stack([x, y])
        
        # Add small random perturbation for diversity
        centers += np.random.normal(0, 0.003, size=(N, 2))
        centers = np.clip(centers, -0.49, 0.49)
        
        # Force-based optimization with momentum
        velocity = np.zeros_like(centers)
        learning_rate = 0.008
        decay_rate = 0.997
        momentum = 0.85
        cutoff = 0.25
        
        current_best_radius = calculate_radius(centers)
        if current_best_radius > best_radius:
            best_radius = current_best_radius
            best_centers = centers.copy()
        
        for step in range(steps_per_restart):
            if time.time() - start_time > time_limit:
                break
            
            # Compute pairwise distances and forces
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist_matrix = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist_matrix, np.inf)
            
            # Only consider pairs within cutoff
            mask = dist_matrix < cutoff
            mask = np.where(mask, 1.0, 0.0)
            np.fill_diagonal(mask, 0.0)
            
            dist_safe = np.where(dist_matrix > 1e-10, dist_matrix, 1e-10)
            
            # Repulsive force: 1/d^3 for closer pairs
            factor = np.zeros_like(dist_matrix)
            factor[mask > 0] = 1.0 / (dist_safe[mask > 0] ** 3)
            
            pair_forces = np.sum(diff * factor[:, :, np.newaxis], axis=1)
            
            # Boundary forces
            d_right = 0.5 - centers[:, 0]
            d_left = 0.5 + centers[:, 0]
            d_top = 0.5 - centers[:, 1]
            d_bottom = 0.5 + centers[:, 1]
            
            f_x = np.zeros(N)
            f_y = np.zeros(N)
            
            mask_r = d_right < cutoff
            f_x[mask_r] -= 1.0 / (np.maximum(d_right[mask_r], 1e-10) ** 2)
            
            mask_l = d_left < cutoff
            f_x[mask_l] += 1.0 / (np.maximum(d_left[mask_l], 1e-10) ** 2)
            
            mask_t = d_top < cutoff
            f_y[mask_t] -= 1.0 / (np.maximum(d_top[mask_t], 1e-10) ** 2)
            
            mask_b = d_bottom < cutoff
            f_y[mask_b] += 1.0 / (np.maximum(d_bottom[mask_b], 1e-10) ** 2)
            
            boundary_forces = np.column_stack([f_x, f_y])
            
            total_forces = pair_forces + boundary_forces
            
            # Normalize forces
            force_norms = np.linalg.norm(total_forces, axis=1)
            max_norm = np.max(force_norms)
            if max_norm > 1e-10:
                total_forces = (total_forces / max_norm) * learning_rate
            else:
                total_forces = np.zeros_like(centers)
            
            # Add small jitter to escape local minima
            jitter = 0.0005 * (1 - step / steps_per_restart)
            if jitter > 0:
                total_forces += np.random.normal(0, jitter, size=(N, 2))
            
            # Update with momentum
            velocity = momentum * velocity + (1 - momentum) * total_forces
            centers += velocity
            centers = np.clip(centers, -0.49999, 0.49999)
            
            # Decay learning rate
            learning_rate *= decay_rate
            
            # Periodically check radius
            if step % 40 == 0:
                current_radius = calculate_radius(centers)
                if current_radius > best_radius:
                    best_radius = current_radius
                    best_centers = centers.copy()
        
        # Final check after each restart
        final_radius = calculate_radius(centers)
        if final_radius > best_radius:
            best_radius = final_radius
            best_centers = centers.copy()
    
    return best_centers
