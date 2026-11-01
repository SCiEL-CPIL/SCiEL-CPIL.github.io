"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin
    
    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Repulsive Force Optimization
    # We want to maximize the minimum distance between points, which correlates with minimizing covering radius.
    # Force F_i = sum_{j != i} (P_i - P_j) / ||P_i - P_j||^2
    # This is the gradient of the potential energy U = sum 1/||P_i - P_j||
    
    step_size = 0.01
    cooling = 0.999
    
    while time.time() - start_time < time_limit:
        # Calculate forces
        forces = np.zeros_like(pts)
        # Vectorized distance calculation for O(N^2)
        # diff[i,j] = pts[i] - pts[j]
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        
        # Avoid division by zero
        np.fill_diagonal(dist_sq, 1.0)
        
        # Force term: (P_i - P_j) / dist^2
        # We want to sum over j for each i
        # weights[i,j] = 1 / dist_sq[i,j]
        weights = 1.0 / dist_sq
        np.fill_diagonal(weights, 0.0)
        
        # forces[i] = sum_j weights[i,j] * diff[i,j]
        forces = np.sum(weights[:, :, np.newaxis] * diff, axis=1)
        
        # Update points
        pts = pts + step_size * forces
        
        # Renormalize to unit sphere
        norms = np.linalg.norm(pts, axis=1)
        pts = pts / norms[:, np.newaxis]
        
        # Cool down step size
        step_size *= cooling
        
        # Optional: Periodic check of covering radius to potentially adjust strategy
        # But repulsive force is robust.
        
        # If step size is very small, we can break early to save time for evaluation
        if step_size < 1e-6:
            break

    # 3. Final Refinement: 
    # The repulsive force method maximizes packing. 
    # To specifically minimize covering radius, we can check the convex hull facets.
    # The covering radius is the maximum angular distance from the origin to the plane of a facet (if the origin is inside).
    # Actually, the covering radius is the maximum circumradius of the spherical Voronoi cells.
    # A simpler check: Evaluate on a dense grid.
    
    # Let's do a quick local refinement on the worst point.
    # Generate a dense grid of test points
    # Use a Fibonacci spiral with many more points for evaluation
    N_eval = 10000
    i_eval = np.arange(N_eval) + 0.5
    z_eval = 1.0 - 2.0 * i_eval / N_eval
    phi_eval = np.pi * (1.0 + np.sqrt(5.0)) * i_eval
    r_eval = np.sqrt(1.0 - z_eval * z_eval)
    eval_pts = np.column_stack([r_eval * np.cos(phi_eval), r_eval * np.sin(phi_eval), z_eval])
    
    # Find the worst point in eval_pts
    # dist_sq[y, i] = ||eval_pts[y] - pts[i]||^2
    diff_eval = eval_pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq_eval = np.sum(diff_eval ** 2, axis=2)
    min_dist_sq = np.min(dist_sq_eval, axis=1)
    worst_idx = np.argmax(min_dist_sq)
    worst_point = eval_pts[worst_idx]
    
    # If the worst point is significantly far, we can try to move the nearest points towards it?
    # Or just return the current best. The repulsive force usually gets very close to optimal.
    # Given the time constraint, returning the repulsive force result is safe.
    
    return pts

# EVOLVE-BLOCK-END
