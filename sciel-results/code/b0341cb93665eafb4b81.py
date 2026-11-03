"""Improved packing of 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import random


def _score(points):
    """Compute the largest common radius for a given set of centres."""
    if points.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # Check inside square [-0.5, 0.5]^2
    if np.any(np.abs(points) > 0.5 + 1e-12):
        return 0.0
    # Distance to boundary
    dist_bound = 0.5 - np.max(np.abs(points), axis=1)
    r_bound = np.min(dist_bound)
    # Pairwise distances
    # Use vectorized computation for all pairs
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    r_pair = np.min(dists) / 2.0
    return min(r_pair, r_bound)


def _hex_grid_init():
    """Generate a hexagonal packing initialization."""
    # Hexagonal packing: rows offset by half a spacing
    # For 200 circles, try various grid layouts
    # Approximate: sqrt(200) ≈ 14.14, so about 14x14 grid
    # In hex packing, density is higher
    
    # Let's try a rectangular grid with hex offset
    # For hex packing in a square, we can use:
    # Row spacing: s * sqrt(3)/2
    # Column spacing: s
    # Offset every other row by s/2
    
    # We want to fit ~200 points
    # Try different numbers of rows
    best_points = None
    best_r = 0
    
    for n_rows in range(12, 18):
        n_cols = int(np.ceil(N / n_rows))
        # Adjust to get exactly N points
        total = n_rows * n_cols
        if total < N:
            continue
        
        # Spacing
        # In hex packing, horizontal spacing = s, vertical = s*sqrt(3)/2
        # Total width ≈ (n_cols - 0.5) * s (with offset)
        # Total height ≈ (n_rows - 1) * s * sqrt(3)/2
        # We want both to fit in [0.5, 0.5] = 1.0
        # So s <= 1.0 / (n_cols - 0.5) and s * sqrt(3)/2 <= 1.0 / (n_rows - 1)
        
        s1 = 1.0 / (n_cols - 0.5)
        s2 = 1.0 / ((n_rows - 1) * np.sqrt(3) / 2)
        s = min(s1, s2)
        
        points = []
        for row in range(n_rows):
            y = -0.5 + s * np.sqrt(3) / 2 * row
            offset = s / 2 if row % 2 == 1 else 0
            for col in range(n_cols):
                x = -0.5 + s * col + offset
                points.append([x, y])
        
        points = np.array(points[:N])
        
        # Check validity
        if np.any(np.abs(points) > 0.5 + 1e-10):
            continue
        
        r = _score(points)
        if r > best_r:
            best_r = r
            best_points = points.copy()
    
    if best_points is None:
        # Fallback to square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_points = grid[:N]
    
    return best_points


def _local_optimize(points, time_limit):
    """Perform local optimization using gradient-free methods."""
    start_time = time.time()
    current_points = points.copy()
    current_r = _score(current_points)
    
    # Use Nelder-Mead on subsets or direct search
    # Since N=200, direct optimization is expensive
    # Instead, do iterative pairwise repulsion
    
    # Simple iterative relaxation
    for iteration in range(500):
        if time.time() - start_time > time_limit - 5:
            break
        
        # Compute forces: repel if too close, attract to center
        improved = False
        for i in range(N):
            # Distance to boundary
            dist_to_bound = 0.5 - np.max(np.abs(current_points[i]))
            
            # Find nearest neighbor
            diffs = current_points - current_points[i]
            dists = np.sqrt(np.sum(diffs ** 2, axis=1))
            dists[i] = np.inf
            min_dist = np.min(dists)
            
            # Current radius constraint for point i
            r_i = min(dist_to_bound, min_dist / 2)
            
            # If this point is limiting, try to move it
            # Direction: away from nearest neighbor and toward center
            nearest_idx = np.argmin(dists)
            dir_away = current_points[i] - current_points[nearest_idx]
            dir_away = dir_away / (np.linalg.norm(dir_away) + 1e-10)
            
            # Also move toward center if near boundary
            dir_center = -current_points[i]
            dir_center = dir_center / (np.linalg.norm(dir_center) + 1e-10)
            
            # Combined direction
            direction = dir_away + 0.5 * dir_center
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            
            # Small step
            step = 0.001
            new_point = current_points[i] + step * direction
            
            # Check if valid
            if np.all(np.abs(new_point) < 0.5):
                old_point = current_points[i].copy()
                current_points[i] = new_point
                new_r = _score(current_points)
                if new_r >= current_r - 1e-12:
                    current_r = new_r
                    improved = True
                else:
                    current_points[i] = old_point
        
        if not improved:
            break
    
    return current_points, current_r


def construct():
    """Construct centres of 200 equal circles inside the unit square."""
    deadline = time.time() + 145  # Leave 5s safety margin
    
    # Start with hex grid initialization
    points = _hex_grid_init()
    current_r = _score(points)
    
    # Local optimization
    if time.time() < deadline - 10:
        points, current_r = _local_optimize(points, deadline - time.time() - 5)
    
    # Final check
    if not np.all(np.isfinite(points)):
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        points = np.array([(x, y) for y in ticks for x in ticks])[:N]
    
    return points
# EVOLVE-BLOCK-END
