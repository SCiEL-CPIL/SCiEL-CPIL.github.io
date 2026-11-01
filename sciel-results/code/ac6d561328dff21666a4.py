"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time

def _normalize(v):
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return v / norms

def construct():
    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = _normalize(points)

    # Pre-compute a grid on the sphere for evaluating covering radius
    # Higher resolution grid for better accuracy
    n_lat = 180
    n_lon = 360
    lats = np.linspace(-np.pi/2, np.pi/2, n_lat, endpoint=False) + np.pi/n_lat
    lons = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
    
    LATS, LONS = np.meshgrid(lats, lons, indexing='ij')
    
    grid_x = np.cos(LATS) * np.cos(LONS)
    grid_y = np.cos(LATS) * np.sin(LONS)
    grid_z = np.sin(LATS)
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
    grid_points = _normalize(grid_points)
    
    start_time = time.time()
    time_limit = 115.0
    
    step_size = 0.02
    patience = 0
    max_patience = 300
    prev_radius = np.pi
    
    while time.time() - start_time < time_limit:
        # 1. Evaluate covering radius on the grid
        # Dots: (G, N)
        dots = np.dot(grid_points, points.T) 
        dots = np.clip(dots, -1.0, 1.0)
        max_dots = np.max(dots, axis=1)
        min_dists = np.arccos(max_dots)
        
        current_radius = np.max(min_dists)
        
        if current_radius < prev_radius - 1e-10:
            prev_radius = current_radius
            patience = 0
        else:
            patience += 1
            if patience > max_patience:
                break
        
        # Find the grid point with the largest distance (the "hole")
        worst_idx = np.argmax(min_dists)
        worst_point = grid_points[worst_idx]
        
        # Find the K nearest points to the worst_point
        # K=4 is often better than 3 for stability
        K = 4
        dots_to_worst = np.dot(points, worst_point)
        nearest_indices = np.argsort(-dots_to_worst)[:K]
        
        # Move these points away from the worst_point
        # We calculate the tangential direction for each
        # Vector from point to worst point
        # We want to move the point AWAY from the worst point to increase the distance?
        # No, if there is a hole, the points are too far from the hole center.
        # We should move the points TOWARDS the hole center to cover it better?
        # Wait. The covering radius is the max distance from ANY point on sphere to NEAREST config point.
        # If 'worst_point' is far from all config points, we need to bring some config points CLOSER to it.
        # So we move the nearest config points TOWARDS the worst_point.
        
        for idx in nearest_indices:
            p = points[idx]
            # Direction from p to worst_point
            vec = worst_point - p
            # Project onto tangent space at p
            # t = vec - (vec . p) p
            proj = vec - np.dot(vec, p) * p
            t_norm = np.linalg.norm(proj)
            if t_norm > 1e-9:
                t = proj / t_norm
                # Move towards worst_point
                points[idx] = _normalize(p + step_size * t)
            else:
                # If they are the same point, push them apart slightly? 
                # Or just skip.
                pass
        
        # Apply a global repulsive force to prevent clumping
        # Vectorized version
        # Compute all pairwise distances
        # This is O(N^2) memory and compute, but N=130 is small enough.
        # dists[i, j] = ||pi - pj||
        
        # Use broadcasting
        # pi: (N, 1, 3), pj: (1, N, 3)
        pi = points[:, np.newaxis, :]
        pj = points[np.newaxis, :, :]
        
        # vec = pi - pj (N, N, 3)
        vec = pi - pj
        # d2 = sum(vec^2, axis=2) (N, N)
        d2 = np.sum(vec * vec, axis=2)
        
        # Avoid division by zero
        d2_safe = np.where(d2 < 1e-12, 1.0, d2)
        
        # Force magnitude: k / d^2
        # We want repulsion. Force on i from j is along (pi - pj).
        # F_i = sum_j (pi - pj) / d_ij^2
        # This is equivalent to sum_j vec[i, j] / d2[i, j]
        
        # Normalize direction and scale
        # vec is already direction.
        # We need vec / d2.
        # Note: vec is (N, N, 3). d2 is (N, N).
        # We want to sum over j for each i.
        
        # F_i = sum_j vec[i, j] / d2[i, j]
        # We can do this by multiplying vec by 1/d2 and summing axis=1.
        
        inv_d2 = 1.0 / d2_safe
        # Set diagonal to 0 to avoid self-interaction
        np.fill_diagonal(inv_d2, 0.0)
        
        # Scale vec
        # vec * inv_d2[..., np.newaxis]
        forces = np.sum(vec * inv_d2[:, :, np.newaxis], axis=1)
        
        # Project forces onto tangent space to keep points on sphere
        # F_tang = F - (F . p) p
        # F: (N, 3), p: (N, 3)
        F_dot_p = np.sum(forces * points, axis=1, keepdims=True)
        tangent_forces = forces - F_dot_p * points
        
        # Apply small step
        points = _normalize(points + 0.001 * tangent_forces)

    return points
# EVOLVE-BLOCK-END
