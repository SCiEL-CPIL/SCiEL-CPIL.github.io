"""Rewritten program for packing 200 equal circles in a unit square."""
import numpy as np
from scipy.optimize import minimize
import time
import random

N = 200

# EVOLVE-BLOCK-START

def _distance_matrix(pts):
    """Compute pairwise distances between points."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))

def _compute_radius(pts):
    """Compute the maximum radius for given centers."""
    # Minimum pairwise distance
    dm = _distance_matrix(pts)
    # Exclude diagonal
    np.fill_diagonal(dm, np.inf)
    min_pair = np.min(dm)
    
    # Distance to boundary: for square [-0.5, 0.5]^2
    # For each point, min distance to any of the 4 sides
    dist_to_bounds = np.minimum(
        np.minimum(0.5 - pts[:, 0], pts[:, 0] + 0.5),
        np.minimum(0.5 - pts[:, 1], pts[:, 1] + 0.5)
    )
    min_bound = np.min(dist_to_bounds)
    
    return min(min_pair, 2 * min_bound) / 2

def _smooth_objective(pts, N=N):
    """
    Smooth objective function to minimize.
    Minimizing the sum of inverse squared distances encourages points to spread out.
    This avoids the non-smoothness of the min-radius objective.
    """
    if not np.all(np.isfinite(pts)):
        return 1e10
    
    # Ensure points are within bounds to avoid huge values from boundary distance
    # We add a small penalty for being outside, but the main term is repulsion
    pts_clipped = np.clip(pts, -0.5, 0.5)
    
    # Pairwise repulsion
    diff = pts_clipped[:, np.newaxis, :] - pts_clipped[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dist_sq, 1.0) # Avoid division by zero
    
    # Use inverse squared distance
    # Clip to avoid infinity if points coincide
    dist_sq_clipped = np.maximum(dist_sq, 1e-8)
    pair_energy = np.sum(1.0 / dist_sq_clipped)
    
    # Boundary repulsion
    # Distance to boundary for each point
    dist_x = np.minimum(0.5 - pts_clipped[:, 0], pts_clipped[:, 0] + 0.5)
    dist_y = np.minimum(0.5 - pts_clipped[:, 1], pts_clipped[:, 1] + 0.5)
    dist_b = np.minimum(dist_x, dist_y)
    dist_b_clipped = np.maximum(dist_b, 1e-8)
    bound_energy = np.sum(1.0 / dist_b_clipped**2)
    
    return pair_energy + bound_energy

def _hex_initialization(rotation, offset_x, offset_y):
    """
    Generate a hexagonal lattice initialization.
    """
    # Approximate spacing for N=200 in unit square
    # Area per point ~ 1/200 = 0.005. Hex area per point = s^2 * sqrt(3)/2.
    # s^2 = 0.005 * 2 / sqrt(3) ~ 0.00577 -> s ~ 0.076
    # Let's use a slightly larger spacing to ensure we have enough points to select from
    s = 0.075
    
    # Generate grid points
    # Hex grid: (i, j) -> (s * (i + 0.5*(j%2)), s * j * sqrt(3)/2)
    # We need to cover [-0.5, 0.5]
    # Range of i, j
    max_idx = int(1.0 / s) + 5
    
    pts = []
    for j in range(-max_idx, max_idx + 1):
        for i in range(-max_idx, max_idx + 1):
            x = s * (i + 0.5 * (j % 2))
            y = s * j * np.sqrt(3) / 2
            
            # Rotate
            xr = x * np.cos(rotation) - y * np.sin(rotation)
            yr = x * np.sin(rotation) + y * np.cos(rotation)
            
            # Offset
            xr += offset_x
            yr += offset_y
            
            # Check if inside square
            if -0.5 <= xr <= 0.5 and -0.5 <= yr <= 0.5:
                pts.append([xr, yr])
    
    pts = np.array(pts)
    
    # If we have more than N points, randomly select N
    if len(pts) > N:
        indices = np.random.choice(len(pts), N, replace=False)
        pts = pts[indices]
    elif len(pts) < N:
        # If we have fewer, pad with random points (shouldn't happen with s=0.075)
        # But for safety
        missing = N - len(pts)
        random_pts = np.random.uniform(-0.5, 0.5, size=(missing, 2))
        pts = np.vstack([pts, random_pts])
        
    return pts

def _local_optimize_smooth(pts, max_iter=1000):
    """
    Optimize using L-BFGS-B on the smooth objective.
    """
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    def obj(x):
        pts_x = x.reshape(-1, 2)
        return _smooth_objective(pts_x)
    
    x0 = pts.flatten()
    
    try:
        result = minimize(
            obj, x0, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-6}
        )
        best_pts = result.x.reshape(-1, 2)
        best_pts = np.clip(best_pts, -0.5, 0.5)
        return best_pts
    except Exception:
        return pts

def construct():
    """Construct the optimal packing of 200 circles."""
    start_time = time.time()
    time_limit = 110  # Leave some margin
    
    best_pts = None
    best_r = 0.0
    
    # Strategy 1: Hexagonal initializations with random rotations/offsets
    # This provides a strong global structure
    n_restarts = 20
    
    for i in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Random rotation and small offset
        rotation = np.random.uniform(0, 2 * np.pi)
        offset_x = np.random.uniform(-0.02, 0.02)
        offset_y = np.random.uniform(-0.02, 0.02)
        
        pts = _hex_initialization(rotation, offset_x, offset_y)
        
        # Optimize
        opt_pts = _local_optimize_smooth(pts, max_iter=500)
        
        # Evaluate true radius
        r = _compute_radius(opt_pts)
        
        if r > best_r:
            best_r = r
            best_pts = opt_pts.copy()
            
    # Strategy 2: Polish the best solution with the non-smooth objective
    # This can squeeze out the last bit of radius
    if best_pts is not None and time.time() - start_time < time_limit - 10:
        
        def _non_smooth_objective(pts, N=N):
            """Objective to minimize: negative of the radius."""
            if not np.all(np.isfinite(pts)):
                return 1e6
            if np.any(pts[:, 0] < -0.5) or np.any(pts[:, 0] > 0.5) or \
               np.any(pts[:, 1] < -0.5) or np.any(pts[:, 1] > 0.5):
                outside = (np.abs(pts[:, 0]) > 0.5) | (np.abs(pts[:, 1]) > 0.5)
                penalty = np.sum(np.maximum(0, np.abs(pts[:, 0]) - 0.5 + 1e-4)) + \
                          np.sum(np.maximum(0, np.abs(pts[:, 1]) - 0.5 + 1e-4))
                return 1e6 + penalty
            
            r = _compute_radius(pts)
            return -r

        bounds = [(-0.5, 0.5)] * (2 * N)
        
        def obj(x):
            pts_x = x.reshape(-1, 2)
            return _non_smooth_objective(pts_x)
        
        x0 = best_pts.flatten()
        
        try:
            result = minimize(
                obj, x0, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            polished_pts = result.x.reshape(-1, 2)
            polished_pts = np.clip(polished_pts, -0.5, 0.5)
            polished_r = _compute_radius(polished_pts)
            
            if polished_r > best_r:
                best_r = polished_r
                best_pts = polished_pts
        except Exception:
            pass

    # Ensure valid output
    if best_pts is None:
        # Fallback: simple grid
        m = 14
        n_rows = int(np.ceil(N / m))
        pts = []
        for i in range(n_rows):
            for j in range(m):
                if len(pts) >= N:
                    break
                x = (j + 0.5) / m - 0.5
                y = (i + 0.5) / n_rows - 0.5
                pts.append([x, y])
        best_pts = np.array(pts[:N])
    
    # Final validation and clipping
    best_pts = np.clip(best_pts, -0.5, 0.5)
    
    return best_pts

# EVOLVE-BLOCK-END
