"""Improved program for packing 200 equal circles in a unit square."""
import numpy as np
import time
import random

N = 200

# EVOLVE-BLOCK-START

def _compute_radius_fast(pts):
    """Compute the maximum radius efficiently."""
    # Use vectorized computation for pairwise distances
    # Only compute upper triangle to save time
    n = len(pts)
    
    # Compute all pairwise squared distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    
    # Set diagonal to inf
    np.fill_diagonal(dist_sq, np.inf)
    
    min_pair_sq = np.min(dist_sq)
    min_pair = np.sqrt(min_pair_sq)
    
    # Distance to boundary
    dx = np.minimum(0.5 - pts[:, 0], pts[:, 0] + 0.5)
    dy = np.minimum(0.5 - pts[:, 1], pts[:, 1] + 0.5)
    dist_to_bounds = np.minimum(dx, dy)
    min_bound = np.min(dist_to_bounds)
    
    return min(min_pair, 2 * min_bound) / 2

def _grid_initialization(rows, cols):
    """Create a grid initialization with given rows and cols."""
    pts = []
    for i in range(rows):
        for j in range(cols):
            if len(pts) >= N:
                break
            x = (j + 0.5) / cols - 0.5
            y = (i + 0.5) / rows - 0.5
            pts.append([x, y])
    pts = np.array(pts[:N])
    return pts

def _local_search(pts, time_limit=5.0):
    """Fast local search using coordinate descent with adaptive step size."""
    start_time = time.time()
    current_pts = pts.copy()
    current_r = _compute_radius_fast(current_pts)
    
    # Adaptive step size
    step_size = 0.02
    no_improve_count = 0
    max_no_improve = 50
    
    directions = np.array([
        [1, 0], [-1, 0], [0, 1], [0, -1],
        [0.707, 0.707], [-0.707, 0.707], [0.707, -0.707], [-0.707, -0.707]
    ])
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try moving each point
        for idx in range(N):
            if time.time() - start_time >= time_limit:
                    break
            
            for d in directions:
                new_pts = current_pts.copy()
                new_pts[idx] = np.clip(current_pts[idx] + step_size * d, -0.5, 0.5)
                
                new_r = _compute_radius_fast(new_pts)
                if new_r > current_r + 1e-12:
                    current_pts = new_pts
                    current_r = new_r
                    improved = True
                    no_improve_count = 0
                    break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                # Reduce step size
                step_size *= 0.7
                if step_size < 1e-6:
                    break
                no_improve_count = 0
            else:
                break
    
    return current_pts, current_r

def _perturb(pts, sigma=0.01):
    """Add random perturbation to escape local minima."""
    return np.clip(pts + np.random.normal(0, sigma, size=pts.shape), -0.5, 0.5)

def construct():
    """Construct the optimal packing of 200 circles."""
    start_time = time.time()
    time_limit = 115  # Leave margin for final validation
    
    best_pts = None
    best_r = 0.0
    
    # Try multiple grid configurations
    grid_configs = [
        (14, 15), (15, 14), (13, 16), (16, 13),
        (14, 14), (15, 15), (12, 17), (17, 12),
        (10, 20), (20, 10), (11, 18), (18, 11)
    ]
    
    # Phase 1: Evaluate grids and pick best starting points
    grid_results = []
    for rows, cols in grid_configs:
        if time.time() - start_time > time_limit - 60:
            break
        pts = _grid_initialization(rows, cols)
        if len(pts) != N:
            continue
        r = _compute_radius_fast(pts)
        grid_results.append((r, pts.copy()))
    
    # Sort by radius (descending)
    grid_results.sort(key=lambda x: -x[0])
    
    # Phase 2: Local search from top grid configurations
    top_k = min(5, len(grid_results))
    for i in range(top_k):
        if time.time() - start_time > time_limit - 10:
            break
        
        r, pts = grid_results[i]
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
        
        # Run local search with time allocation
        time_per_search = max(5.0, (time_limit - 10 - (time.time() - start_time)) / (top_k - i))
        time_per_search = min(time_per_search, 15.0)
        
        opt_pts, opt_r = _local_search(pts, time_limit=time_per_search)
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts
    
    # Phase 3: Random restarts with perturbation from best solution
    if best_pts is not None:
        remaining_time = time_limit - 5 - (time.time() - start_time)
        if remaining_time > 10:
            n_restarts = int(remaining_time / 8.0)
            n_restarts = max(1, min(n_restarts, 15))
            
            for _ in range(n_restarts):
                if time.time() - start_time > time_limit - 5:
                    break
                
                # Perturb the best solution
                perturbed = _perturb(best_pts, sigma=0.015)
                
                # Run local search
                opt_pts, opt_r = _local_search(perturbed, time_limit=6.0)
                if opt_r > best_r:
                    best_r = opt_r
                    best_pts = opt_pts
    
    # Phase 4: Final refinement
    if best_pts is not None and time.time() - start_time < time_limit - 3:
        remaining = time_limit - 3 - (time.time() - start_time)
        if remaining > 2:
            final_pts, final_r = _local_search(best_pts, time_limit=remaining)
            if final_r > best_r:
                best_r = final_r
                best_pts = final_pts
    
    # Ensure valid output
    if best_pts is None:
        # Fallback
        pts = _grid_initialization(14, 15)
        best_pts = pts
    
    # Final validation and clipping
    best_pts = np.clip(best_pts, -0.5, 0.5)
    
    return best_pts

# EVOLVE-BLOCK-END
