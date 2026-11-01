"""Optimized program for covering the sphere with 130 equal caps."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import time


def _spherical_grid(n_points=10000):
    """Generate a dense spherical grid using Fibonacci spiral."""
    i = np.arange(n_points) + 0.5
    z = 1.0 - 2.0 * i / n_points
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _covering_radius_exact(points, grid):
    """Compute the exact covering radius in radians using a dense grid."""
    tree = cKDTree(points)
    dists, _ = tree.query(grid)
    max_d = np.max(dists)
    if max_d >= 2.0:
        return np.pi
    return 2.0 * np.arcsin(np.clip(max_d / 2.0, -1.0, 1.0))


def _smooth_objective(x, grid, temperature=0.1):
    """
    Smooth, differentiable objective function using soft-min of squared distances.
    """
    pts = x.reshape(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    pts = pts / norms
    
    # Compute squared Euclidean distances from each grid point to each sphere point
    dots = grid @ pts.T  # (M, N)
    dist_sq = 2.0 - 2.0 * dots  # (M, N)
    dist_sq = np.maximum(dist_sq, 0.0)
    
    # Soft min over points for each grid point
    scaled = -dist_sq / temperature
    max_scaled = np.max(scaled, axis=1, keepdims=True)
    log_sum_exp = max_scaled.squeeze() + np.log(np.sum(np.exp(scaled - max_scaled), axis=1))
    soft_min_dist_sq = -temperature * log_sum_exp
    
    # Smooth max over grid points
    temp2 = temperature * 0.5
    scaled2 = -soft_min_dist_sq / temp2
    max_scaled2 = np.max(scaled2)
    log_sum_exp2 = max_scaled2 + np.log(np.sum(np.exp(scaled2 - max_scaled2)))
    smooth_max_dist_sq = -temp2 * log_sum_exp2
    
    d = np.sqrt(np.maximum(smooth_max_dist_sq, 0.0))
    if d >= 2.0:
        return np.pi
    return 2.0 * np.arcsin(np.clip(d / 2.0, -1.0, 1.0))


def _initial_points():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def construct():
    """Construct 130 points on the unit sphere minimizing the covering radius."""
    start_time = time.time()
    timeout = 115.0  # Leave 5s buffer
    
    # Initial points
    points = _initial_points()
    
    # Phase 1: Coarse optimization with moderate grid
    grid_coarse = _spherical_grid(n_points=20000)
    best_points = points.copy()
    best_radius = _covering_radius_exact(points, grid_coarse)
    
    try:
        # Annealing schedule with increasing grid density
        # (temperature, grid_size)
        phases = [
            (0.5, 20000),
            (0.2, 20000),
            (0.1, 50000),
            (0.05, 50000),
            (0.02, 100000),
            (0.01, 100000),
        ]
        
        current_grid_size = 0
        grid = None
        
        for temp, grid_size in phases:
            if time.time() - start_time > timeout:
                break
            
            # Generate new grid if size changed
            if grid_size != current_grid_size:
                grid = _spherical_grid(n_points=grid_size)
                current_grid_size = grid_size
            
            def obj_smooth(x):
                return _smooth_objective(x, grid, temperature=temp)
            
            x0 = best_points.flatten()
            
            result = minimize(
                obj_smooth,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            if time.time() - start_time > timeout:
                break
            
            opt_points = result.x.reshape(N, 3)
            norms = np.linalg.norm(opt_points, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            opt_points = opt_points / norms
            
            # Evaluate on a slightly denser grid to check actual improvement
            # Cap eval grid size to avoid excessive memory/time
            eval_grid_size = min(int(grid_size * 1.5), 200000)
            eval_grid = _spherical_grid(n_points=eval_grid_size)
            opt_radius = _covering_radius_exact(opt_points, eval_grid)
            
            if opt_radius < best_radius:
                best_points = opt_points
                best_radius = opt_radius

    except Exception:
        pass
    
    # Final validation and normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    best_points = best_points / norms
    
    # Check for coincident points
    tree = cKDTree(best_points)
    dists, _ = tree.query(best_points, k=2)
    if np.min(dists[:, 1]) < 1e-6:
        best_points = _initial_points()
    
    return best_points
# EVOLVE-BLOCK-END
