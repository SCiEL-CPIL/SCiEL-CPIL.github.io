"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START

def _compute_radius(pts):
    """Compute the maximum common radius for given centers."""
    if len(pts) < 2:
        return 0.0
    # Min distance between any pair
    from scipy.spatial.distance import cdist
    dist = cdist(pts, pts)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    # Min distance to boundary
    dx = 0.5 - np.abs(pts[:, 0])
    dy = 0.5 - np.abs(pts[:, 1])
    min_boundary = np.min(np.minimum(dx, dy))
    return min(min_dist / 2.0, min_boundary)

def _initial_7x7_plus_1():
    """Create a 7x7 grid (49 points) and add one point at the center."""
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    pts = np.array([(x, y) for y in ticks for x in ticks])
    # Add center point
    pts = np.vstack([pts, [[0.0, 0.0]]])
    return pts[:N].astype(np.float64)

def _smooth_objective(x, power=4):
    """
    Smooth objective function to minimize.
    Minimizes the sum of inverse power distances to encourage separation.
    """
    pts = x.reshape(-1, 2)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Set diagonal to inf to ignore self-distance
    np.fill_diagonal(dist_sq, np.inf)
    
    # Inverse power distances
    # Add epsilon to avoid division by zero
    epsilon = 1e-12
    inv_dist_pow = np.sum(1.0 / (dist_sq + epsilon) ** (power / 2.0)) / 2.0
    
    # Boundary repulsions
    dx = 0.5 - np.abs(pts[:, 0])
    dy = 0.5 - np.abs(pts[:, 1])
    wall_dist = np.minimum(dx, dy)
    
    # If any point is outside, return a large value
    if np.any(wall_dist <= 0):
        return 1e6
    
    wall_repulsion = np.sum(1.0 / (wall_dist + epsilon) ** power)
    
    return inv_dist_pow + wall_repulsion

def _local_search_separation(pts, rng, time_budget=10.0):
    """
    Local search that explicitly targets increasing the minimum distance.
    Identifies the closest pair and closest wall point, and moves them apart.
    """
    start_time = time.time()
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    
    step_size = 0.01
    
    while time.time() - start_time < time_budget:
        # Compute current radius and identify bottlenecks
        from scipy.spatial.distance import cdist
        dist = cdist(best_pts, best_pts)
        np.fill_diagonal(dist, np.inf)
        min_dist_val = np.min(dist)
        min_dist_idx = np.unravel_index(np.argmin(dist), dist.shape)
        
        dx = 0.5 - np.abs(best_pts[:, 0])
        dy = 0.5 - np.abs(best_pts[:, 1])
        wall_dists = np.minimum(dx, dy)
        min_wall_val = np.min(wall_dists)
        min_wall_idx = np.argmin(wall_dists)
        
        current_r = min(min_dist_val / 2.0, min_wall_val)
        
        if current_r < best_r - 1e-9:
            # If we somehow got worse, revert? No, we only accept improvements.
            # This loop only runs if we accepted a move, so this shouldn't happen often.
            break
            
        # Determine if pair distance or wall distance is the bottleneck
        if min_dist_val / 2.0 < min_wall_val:
            # Pair distance is bottleneck
            i, j = min_dist_idx
            # Move i and j apart
            vec = best_pts[i] - best_pts[j]
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec = vec / norm
            else:
                vec = rng.normal(size=2)
                vec = vec / np.linalg.norm(vec)
            
            move_i = vec * step_size
            move_j = -vec * step_size
            
            new_pts = best_pts.copy()
            new_pts[i] += move_i
            new_pts[j] += move_j
        else:
            # Wall distance is bottleneck
            i = min_wall_idx
            # Move point i towards center
            vec = -best_pts[i]
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec = vec / norm
            else:
                vec = rng.normal(size=2)
                vec = vec / np.linalg.norm(vec)
            
            move_i = vec * step_size
            new_pts = best_pts.copy()
            new_pts[i] += move_i
        
        # Clip to square
        new_pts = np.clip(new_pts, -0.5, 0.5)
        
        new_r = _compute_radius(new_pts)
        
        if new_r > best_r:
            best_r = new_r
            best_pts = new_pts
            step_size = max(0.001, step_size * 0.95)
        else:
            step_size = min(0.05, step_size * 1.1)
            
            # If step size is very small and no improvement, try random perturbation
            if step_size < 0.002:
                delta = rng.normal(0, 0.005, size=best_pts.shape)
                new_pts = best_pts + delta
                new_pts = np.clip(new_pts, -0.5, 0.5)
                new_r = _compute_radius(new_pts)
                if new_r > best_r:
                    best_r = new_r
                    best_pts = new_pts
                    step_size = 0.01
                else:
                    break

    return best_pts, best_r

def construct():
    """Construct centers of N equal circles inside the square [-1/2, 1/2]^2."""
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    # Start with 7x7 + 1 initialization
    pts = _initial_7x7_plus_1()
    best_radius = _compute_radius(pts)
    best_pts = pts.copy()
    
    rng = np.random.default_rng(42)
    
    # Strategy 1: Gradient optimization using smooth objective
    try:
        if time.time() - start_time < time_limit - 40:
            x0 = pts.flatten().astype(np.float64)
            bounds = [(-0.5, 0.5)] * (2 * N)
            
            # Use L-BFGS-B
            res = minimize(_smooth_objective, x0, method='L-BFGS-B', 
                           bounds=bounds,
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
            
            new_pts = res.x.reshape(-1, 2)
            new_pts = np.clip(new_pts, -0.5, 0.5)
            new_r = _compute_radius(new_pts)
            
            if new_r > best_radius:
                best_radius = new_r
                best_pts = new_pts
    except Exception:
        pass
    
    # Strategy 2: Local search for separation
    try:
        if time.time() - start_time < time_limit - 30:
            ls_pts, ls_r = _local_search_separation(best_pts, rng, time_budget=20.0)
            if ls_r > best_radius:
                best_radius = ls_r
                best_pts = ls_pts
    except Exception:
        pass
    
    # Strategy 3: Random restarts with quick local search
    try:
        remaining_time = time_limit - (time.time() - start_time) - 10
        if remaining_time > 10:
            n_restarts = int(remaining_time / 5)
            n_restarts = max(1, min(n_restarts, 5))
            
            for i in range(n_restarts):
                if time.time() - start_time > time_limit - 10:
                    break
                
                # Perturb the best solution
                rand_pts = best_pts + rng.normal(0, 0.02, size=best_pts.shape)
                rand_pts = np.clip(rand_pts, -0.5, 0.5)
                
                # Quick local search
                ls_pts, ls_r = _local_search_separation(rand_pts, rng, time_budget=3.0)
                
                if ls_r > best_radius:
                    best_radius = ls_r
                    best_pts = ls_pts
    except Exception:
        pass
    
    # Final polish
    try:
        if time.time() - start_time < time_limit - 5:
            final_pts, final_r = _local_search_separation(best_pts, rng, time_budget=5.0)
            if final_r > best_radius:
                best_radius = final_r
                best_pts = final_pts
    except Exception:
        pass
    
    return best_pts

# EVOLVE-BLOCK-END
