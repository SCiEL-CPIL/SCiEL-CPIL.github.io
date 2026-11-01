# EVOLVE-BLOCK-START
import numpy as np
import time

def _normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def construct():
    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Pre-compute a grid on the sphere for evaluating covering radius
    # Higher resolution for better accuracy
    n_lat = 200
    n_lon = 400
    lats = np.linspace(-np.pi/2, np.pi/2, n_lat, endpoint=False) + np.pi/n_lat
    lons = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
    
    LATS, LONS = np.meshgrid(lats, lons, indexing='ij')
    
    grid_x = np.cos(LATS) * np.cos(LONS)
    grid_y = np.cos(LATS) * np.sin(LONS)
    grid_z = np.sin(LATS)
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
    grid_points = grid_points / np.linalg.norm(grid_points, axis=1, keepdims=True)
    
    start_time = time.time()
    time_limit = 110.0
    
    step_size = 0.05
    repulsive_strength = 0.01
    patience = 0
    max_patience = 100
    prev_radius = np.pi
    
    while time.time() - start_time < time_limit:
        # 1. Evaluate covering radius on the grid
        dots = np.dot(grid_points, points.T) 
        dots = np.clip(dots, -1.0, 1.0)
        max_dots = np.max(dots, axis=1) 
        min_dists = np.arccos(max_dots) 
        
        current_radius = np.max(min_dists)
        
        if current_radius < prev_radius - 1e-6:
            prev_radius = current_radius
            patience = 0
        else:
            patience += 1
            if patience > max_patience:
                break
        
        # Find the grid point with the largest distance (the "hole")
        worst_idx = np.argmax(min_dists)
        worst_point = grid_points[worst_idx]
        
        # Find the 3 nearest points to the worst_point
        dots_to_worst = np.dot(points, worst_point)
        nearest_indices = np.argsort(-dots_to_worst)[:3]
        
        # Move these 3 points TOWARDS the worst_point to fill the hole
        for idx in nearest_indices:
            p = points[idx]
            proj = worst_point - np.dot(worst_point, p) * p
            t = _normalize(proj)
            # Move towards worst_point
            points[idx] = _normalize(p + step_size * t)
            
        # Apply global repulsive force to maintain uniformity
        # Vectorized repulsion calculation for speed
        # Calculate pairwise distances
        # dist2[i,j] = |pi - pj|^2 = 2 - 2*pi.pj
        dots_all = np.dot(points, points.T)
        dist2 = 2.0 - 2.0 * dots_all
        np.fill_diagonal(dist2, 1.0) # Avoid division by zero, will be masked
        
        # Force magnitude: 1 / d^2 = 1 / dist2
        # Direction: tangential component of (pi - pj)
        # vec = pi - pj
        # tang = vec - (vec.pi)pi
        # We can compute the force on each point i by summing contributions from j
        
        forces = np.zeros_like(points)
        for i in range(N):
            pi = points[i]
            # Tangential component of (pi - pj) for all j
            # vec = pi[None, :] - points
            vec = pi[None, :] - points
            # dot(vec, pi)
            dot_vec_pi = np.dot(vec, pi)
            # tang = vec - dot_vec_pi[:, None] * pi[None, :]
            tang = vec - dot_vec_pi[:, None] * pi[None, :]
            
            # Magnitude
            mag = repulsive_strength / np.maximum(dist2[i], 1e-6)
            mag[i] = 0.0 # No self-force
            
            # Force contribution = (tang / |tang|) * mag ? 
            # Actually, the repulsive force should be along the tangent direction away from pj.
            # The vector vec = pi - pj points from pj to pi.
            # The tangential component tang points in the same general direction.
            # We normalize tang to get the unit direction.
            tang_norms = np.linalg.norm(tang, axis=1)
            tang_norms[i] = 1.0 # Avoid div by zero
            unit_tang = tang / tang_norms[:, None]
            
            forces[i] = np.sum(unit_tang * mag[:, None], axis=0)
            
            if np.linalg.norm(forces[i]) > 1e-6:
                points[i] = _normalize(pi + forces[i])
                
        points = points / np.linalg.norm(points, axis=1, keepdims=True)

    return points
# EVOLVE-BLOCK-END
