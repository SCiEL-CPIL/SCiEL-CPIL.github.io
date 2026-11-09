# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 600.0  # 600s limit, keep safety margin

    # 1. Initialize points on a unit sphere using Fibonacci lattice for good distribution
    def fibonacci_sphere(n):
        points = np.zeros((n, 3))
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n):
            z = 1 - (i / (n - 1)) * 2  # z goes from 1 to -1
            radius_at_z = np.sqrt(1 - z * z)
            theta = golden_ratio * i * 2 * np.pi
            x = np.cos(theta) * radius_at_z
            y = np.sin(theta) * radius_at_z
            points[i] = [x, y, z]
        return points

    points = fibonacci_sphere(n)
    
    # Normalize to unit sphere initially
    points /= np.linalg.norm(points, axis=1, keepdims=True)

    best_points = points.copy()
    best_score = -1.0

    def calculate_score(pts):
        if len(pts) < 2:
            return 0.0
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    best_score = calculate_score(best_points)

    # 2. Optimize using repulsion
    # We want to maximize d_min. 
    # Strategy: Move each point away from its nearest neighbors.
    # To avoid local minima, we can use a cooling schedule for step size.
    
    num_iterations = 10000
    base_step_size = 0.1
    cooling_rate = 0.999
    
    for it in range(num_iterations):
        if time.time() > deadline - 30: # Safety margin
            break
            
        step_size = base_step_size * (cooling_rate ** it)
        
        # Calculate forces
        # Force on i from j is (r_i - r_j) / |r_i - r_j|^3
        # This is the gradient of -1/|r_i - r_j|
        # Maximizing min distance is equivalent to minimizing a potential energy 
        # that is very sensitive to close distances.
        
        forces = np.zeros_like(points)
        
        # Vectorized calculation of distances and forces
        # dist_matrix[i,j] = ||points[i] - points[j]||
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Avoid division by zero
        np.fill_diagonal(dists, 1.0)
        
        # Weights: 1/d^2 or similar. 
        # Using 1/d^2 makes the force stronger for closer points, 
        # which helps maximize the minimum distance.
        # F_ij = (r_i - r_j) / d_ij^3
        # We sum these forces for each i.
        
        # dists_inv3 = 1 / dists^3
        dists_inv3 = np.zeros_like(dists)
        mask = dists > 1e-8
        dists_inv3[mask] = 1.0 / (dists[mask] ** 3)
        np.fill_diagonal(dists_inv3, 0.0)
        
        # forces[i] = sum_j (points[i] - points[j]) * dists_inv3[i,j]
        # diff[i,j] is points[i] - points[j]
        forces = np.sum(diff * dists_inv3[:, :, np.newaxis], axis=1)
        
        # Update points
        # Move in direction of force
        points += step_size * forces
        
        # Project back to unit sphere
        # This constraint keeps d_max <= 2 and encourages spherical distribution
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        points /= norms
        
        # Evaluate score
        current_score = calculate_score(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
            
    # Final check
    if time.time() > deadline - 10:
        return best_points

    # Ensure the returned points are valid (finite)
    if not np.all(np.isfinite(best_points)):
        # Fallback to initial fibonacci sphere
        return fibonacci_sphere(n)
        
    return best_points

# EVOLVE-BLOCK-END
