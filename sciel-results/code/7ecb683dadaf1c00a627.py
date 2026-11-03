"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time

def _generate_grid(n_points=10000):
    """Generate a dense set of points on the sphere for discretization."""
    i = np.arange(n_points) + 0.5
    z = 1.0 - 2.0 * i / n_points
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_grid_covering_radius(points, grid):
    """
    Compute the covering radius using the grid approximation.
    Returns the max distance, nearest indices, and dot products.
    """
    dots = grid @ points.T  # (M, N)
    dots = np.clip(dots, -1.0, 1.0)
    
    max_dots = np.max(dots, axis=1) 
    max_dots = np.clip(max_dots, -1.0, 1.0)
    angles = np.arccos(max_dots)
    
    max_angle = np.max(angles)
    nearest_idx = np.argmax(dots, axis=1) 
    
    return max_angle, nearest_idx, dots

def _optimize_grid(points, grid, time_limit):
    """
    Optimize points using grid-based force descent.
    """
    start_time = time.time()
    pts = points.copy()
    M = grid.shape[0]
    N = pts.shape[0]
    
    step_size = 0.05
    best_pts = pts.copy()
    best_radius = _compute_grid_covering_radius(pts, grid)[0]
    
    no_improve = 0
    max_no_improve = 20
    
    while time.time() - start_time < time_limit:
        max_angle, nearest_idx, dots = _compute_grid_covering_radius(pts, grid)
        
        if max_angle < best_radius:
            best_radius = max_angle
            best_pts = pts.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                break
            step_size *= 0.95
            if step_size < 1e-5:
                break

        y = grid[nearest_idx] # (M, 3)
        u = dots[np.arange(M), nearest_idx] # (M,)
        
        denom = np.sqrt(np.maximum(1.0 - u*u, 1e-10))
        
        p_i = pts[nearest_idx]
        dirs = y - p_i * u[:, np.newaxis] # (M, 3)
        
        dirs = dirs / denom[:, np.newaxis]
        
        forces = np.zeros((N, 3))
        np.add.at(forces, nearest_idx, dirs)
        
        counts = np.bincount(nearest_idx, minlength=N)
        counts[counts == 0] = 1
        forces /= counts[:, np.newaxis]
        
        new_pts = pts + step_size * forces
        new_pts /= np.linalg.norm(new_pts, axis=1, keepdims=True)
        
        new_radius = _compute_grid_covering_radius(new_pts, grid)[0]
        
        if new_radius < max_angle:
            pts = new_pts
        else:
            pass

    return best_pts, best_radius

def construct():
    start_time = time.time()
    total_time_limit = 140.0 # Safety margin
    
    grid = _generate_grid(10000)
    
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = fib_pts
    best_radius = np.inf
    
    time_left = total_time_limit - (time.time() - start_time)
    if time_left > 5:
        res_pts, res_rad = _optimize_grid(fib_pts, grid, time_left - 5)
        if res_rad < best_radius:
            best_radius = res_rad
            best_pts = res_pts
            
    return best_pts
# EVOLVE-BLOCK-END
