import numpy as np
import time
from scipy.spatial import cKDTree # Not strictly needed if we do matrix ops, but useful for reference. We will stick to matrix ops for speed on small N.

N = 130

def _generate_grid(num_points):
    """Generate a quasi-uniform grid on the sphere."""
    # Use a Fibonacci spiral for the grid as well, or a more structured approach
    # For covering radius estimation, a dense random or spiral set is good.
    i = np.arange(num_points) + 0.5
    z = 1.0 - 2.0 * i / num_points
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_covering_radius_and_worst(points, grid):
    """
    Compute the covering radius and the worst points on the grid.
    points: (N, 3)
    grid: (M, 3)
    Returns: max_dist, worst_points_indices, nearest_indices
    """
    # Compute dot products: (M, N)
    # grid: (M, 3), points.T: (3, N) -> (M, N)
    dots = grid @ points.T
    # Clip to [-1, 1] for numerical stability
    dots = np.clip(dots, -1.0, 1.0)
    # Angular distance in radians
    dists = np.arccos(dots)
    
    # For each grid point, find the min distance and the index of the nearest point
    min_dists = np.min(dists, axis=1)
    nearest_indices = np.argmin(dists, axis=1)
    
    max_dist = np.max(min_dists)
    
    # Find indices of the worst points (e.g., top 100)
    num_worst = min(100, len(min_dists))
    # Get indices of the largest distances
    worst_indices = np.argsort(min_dists)[::-1][:num_worst]
    
    return max_dist, worst_indices, nearest_indices

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s safety margin
    
    # Initial configuration: Fibonacci spiral
    def initial_config():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = initial_config()
    best_radius = np.inf
    
    # Grid for evaluation
    grid_size = 20000
    grid = _generate_grid(grid_size)
    
    # Optimization parameters
    step_size = 0.1
    momentum = 0.5
    num_worst = 50
    
    # Pre-compute grid once
    
    current_points = best_points.copy()
    velocities = np.zeros_like(current_points)
    
    # Main optimization loop
    while time.time() < deadline:
        # Evaluate
        max_dist, worst_indices, nearest_indices = _compute_covering_radius_and_worst(current_points, grid)
        
        if max_dist < best_radius:
            best_radius = max_dist
            best_points = current_points.copy()
        
        # If we are getting very close to the theoretical limit or stuck, we might break
        # But let's keep going to refine.
        
        # Calculate forces
        forces = np.zeros_like(current_points)
        
        for idx in worst_indices:
            y = grid[idx]
            p_idx = nearest_indices[idx]
            p = current_points[p_idx]
            
            # Vector from p to y
            v = y - p
            # Project v onto the tangent plane at p
            # Tangent plane is orthogonal to p
            v_proj = v - np.dot(v, p) * p
            # Normalize force direction
            norm_v_proj = np.linalg.norm(v_proj)
            if norm_v_proj > 1e-8:
                v_dir = v_proj / norm_v_proj
            else:
                v_dir = v_proj # zero vector
            
            # Add to force with weight based on how bad it is (optional, but simple equal weight works)
            forces[p_idx] += v_dir
        
        # Update velocities with momentum
        velocities = momentum * velocities + step_size * forces
        
        # Update positions
        current_points = current_points + velocities
        
        # Normalize to unit sphere
        norms = np.linalg.norm(current_points, axis=1, keepdims=True)
        current_points = current_points / norms
        
        # Decay step size slowly to allow fine-tuning
        step_size *= 0.999
        
        # Check for convergence (optional, but good to have)
        # if step_size < 1e-6: break

    # Final check: ensure best_points are valid and normalized
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points

# EVOLVE-BLOCK-END
