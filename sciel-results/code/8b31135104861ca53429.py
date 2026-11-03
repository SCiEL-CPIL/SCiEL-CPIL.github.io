import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140 seconds to be safe
    
    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = np.inf
    
    # Parameters for optimization
    learning_rate = 0.01
    max_iterations = 10000
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
            
        # Compute pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, 1.0)  # Avoid division by zero
        
        # Compute repulsive forces (vectorized)
        # Force on point i from point j is proportional to (points[j] - points[i]) / dist^3
        # This gives a 1/r^2 magnitude in the direction away from j
        inv_dists = 1.0 / dists
        inv_dists_cubed = inv_dists**3
        np.fill_diagonal(inv_dists_cubed, 0)  # No self-force
        
        forces = np.sum(diff * inv_dists_cubed[:, :, np.newaxis], axis=1)
        
        # Update points
        points = points + learning_rate * forces
        
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        # Compute covering radius
        hull = ConvexHull(points)
        facets = hull.simplices
        max_radius = 0
        for facet in facets:
            # Get the facet points
            facet_points = points[facet]
            # Compute the circumcenter and radius
            # Using the formula for circumradius of a triangle
            a = np.linalg.norm(facet_points[0] - facet_points[1])
            b = np.linalg.norm(facet_points[1] - facet_points[2])
            c = np.linalg.norm(facet_points[2] - facet_points[0])
            area = 0.5 * np.linalg.norm(np.cross(facet_points[1] - facet_points[0], facet_points[2] - facet_points[0]))
            if area > 1e-10:
                circumradius = (a * b * c) / (4 * area)
                # Convert to angular distance
                angular_radius = np.arccos(np.clip(1 - 2 * circumradius**2, -1, 1)) * 180 / np.pi
                max_radius = max(max_radius, angular_radius)
        
        if max_radius < best_radius:
            best_radius = max_radius
            best_points = points.copy()
        
        # Decay learning rate
        learning_rate *= 0.999
    
    return best_points
