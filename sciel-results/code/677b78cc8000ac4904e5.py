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
    # Increased resolution for better accuracy
    n_lat = 200
    n_lon = 400
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
    
    # Adaptive step size
    step_size = 0.05
    step_decay = 0.999
    min_step_size = 0.001
    
    patience = 0
    max_patience = 200
    prev_radius = np.pi
    
    while time.time() - start_time < time_limit:
        # 1. Evaluate covering radius on the grid
        # Dots: (G, N)
        dots = np.dot(grid_points, points.T) 
        dots = np.clip(dots, -1.0, 1.0)
        max_dots = np.max(dots, axis=1)
        min_dists = np.arccos(max_dots)
        
        current_radius = np.max(min_dists)
        
        if current_radius < prev_radius - 1e-12:
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
        # K=5 for slightly more stability
        K = 5
        dots_to_worst = np.dot(points, worst_point)
        nearest_indices = np.argsort(-dots_to_worst)[:K]
        
        # Move these points towards the worst_point
        for idx in nearest_indices:
            p = points[idx]
            # Direction from p to worst_point
            vec = worst_point - p
            # Project onto tangent space at p
            proj = vec - np.dot(vec, p) * p
            t_norm = np.linalg.norm(proj)
            if t_norm > 1e-9:
                t = proj / t_norm
                # Move towards worst_point
                points[idx] = _normalize(p + step_size * t)
            else:
                pass
        
        # Apply a global repulsive force to prevent clumping
        pi = points[:, np.newaxis, :]
        pj = points[np.newaxis, :, :]
        vec = pi - pj
        d2 = np.sum(vec * vec, axis=2)
        d2_safe = np.where(d2 < 1e-12, 1.0, d2)
        
        inv_d2 = 1.0 / d2_safe
        np.fill_diagonal(inv_d2, 0.0)
        
        forces = np.sum(vec * inv_d2[:, :, np.newaxis], axis=1)
        
        F_dot_p = np.sum(forces * points, axis=1, keepdims=True)
        tangent_forces = forces - F_dot_p * points
        
        # Scale repulsion relative to step size to balance attraction
        repulsion_scale = step_size * 0.1
        points = _normalize(points + repulsion_scale * tangent_forces)
        
        # Decay step size
        step_size = max(min_step_size, step_size * step_decay)

    return points
# EVOLVE-BLOCK-END
