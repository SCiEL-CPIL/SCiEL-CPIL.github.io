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

def _objective(pts, N=N):
    """Objective to minimize: negative of the radius."""
    # Check validity
    if not np.all(np.isfinite(pts)):
        return 1e6
    if np.any(pts[:, 0] < -0.5) or np.any(pts[:, 0] > 0.5) or \
       np.any(pts[:, 1] < -0.5) or np.any(pts[:, 1] > 0.5):
        # Points outside square - heavy penalty
        outside = (np.abs(pts[:, 0]) > 0.5) | (np.abs(pts[:, 1]) > 0.5)
        penalty = np.sum(np.maximum(0, np.abs(pts[:, 0]) - 0.5 + 1e-4)) + \
                  np.sum(np.maximum(0, np.abs(pts[:, 1]) - 0.5 + 1e-4))
        return 1e6 + penalty
    
    r = _compute_radius(pts)
    return -r

def _grid_initialization(m):
    """Create a grid initialization with m columns."""
    n_rows = int(np.ceil(N / m))
    pts = []
    for i in range(n_rows):
        for j in range(m):
            if len(pts) >= N:
                break
            x = (j + 0.5) / m - 0.5
            y = (i + 0.5) / n_rows - 0.5
            pts.append([x, y])
    pts = np.array(pts[:N])
    return pts

def _jittered_grid_initialization(m, jitter=0.01):
    """Create a jittered grid initialization to break symmetry."""
    pts = _grid_initialization(m)
    # Add small random jitter
    jitter_amount = np.random.uniform(-jitter, jitter, size=pts.shape)
    pts = pts + jitter_amount
    # Clip to bounds
    pts = np.clip(pts, -0.5, 0.5)
    return pts

def _random_initialization():
    """Random initialization within the square."""
    return np.random.uniform(-0.5, 0.5, size=(N, 2))

def _jitter(pts, sigma=0.01):
    """Add small random jitter to points."""
    return pts + np.random.normal(0, sigma, size=pts.shape)

def _repulsion_search(pts, max_iter=5000, step_size=0.01, decay=0.999):
    """
    Perform local optimization using a repulsion-based heuristic.
    Moves the closest pair apart and handles boundary constraints.
    """
    current_pts = pts.copy()
    current_r = _compute_radius(current_pts)
    current_step = step_size
    
    for it in range(max_iter):
        if current_step < 1e-7:
            break
            
        # Compute distances
        dm = _distance_matrix(current_pts)
        np.fill_diagonal(dm, np.inf)
        
        # Find closest pair
        min_idx = np.unravel_index(np.argmin(dm), dm.shape)
        i, j = min_idx[0], min_idx[1]
        
        # Move i and j apart
        vec = current_pts[i] - current_pts[j]
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        else:
            vec = np.array([1.0, 0.0])
            
        # Repulsion force
        force = current_step * 0.5
        
        old_i = current_pts[i].copy()
        old_j = current_pts[j].copy()
        
        new_i = current_pts[i] + vec * force
        new_j = current_pts[j] - vec * force
        
        # Check boundaries for i and j
        new_i = np.clip(new_i, -0.5, 0.5)
        new_j = np.clip(new_j, -0.5, 0.5)
        
        # Update points
        current_pts[i] = new_i
        current_pts[j] = new_j
        
        # Compute new radius
        new_r = _compute_radius(current_pts)
        
        if new_r > current_r:
            current_r = new_r
        else:
            # Revert if it didn't improve
            current_pts[i] = old_i
            current_pts[j] = old_j
            # If we revert, we might be stuck. Reduce step size more aggressively?
            # Or try a different pair?
            # For simplicity, we just decay step size.
            pass
        
        current_step *= decay
        
    return current_pts, current_r

def _local_optimize(pts, max_iter=5000, tol=1e-8):
    """Perform local optimization using BFGS."""
    # Use L-BFGS-B with bounds
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    def obj(x):
        pts_x = x.reshape(-1, 2)
        return _objective(pts_x)
    
    x0 = pts.flatten()
    
    try:
        result = minimize(
            obj, x0, method='L-BFGS-B', bounds=bounds,
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-6}
        )
        if result.success or result.status == 1:
            best_pts = result.x.reshape(-1, 2)
            # Ensure points are within bounds
            best_pts = np.clip(best_pts, -0.5, 0.5)
            return best_pts, -_objective(best_pts)
        else:
            return pts, _compute_radius(pts)
    except Exception:
        return pts, _compute_radius(pts)

def _hill_climbing(pts, n_iterations=100, step_size=0.005):
    """Simple hill climbing with random moves."""
    current_pts = pts.copy()
    current_r = _compute_radius(current_pts)
    
    for _ in range(n_iterations):
        # Try a random move
        idx = np.random.randint(0, N)
        move = np.random.normal(0, step_size, size=2)
        new_pts = current_pts.copy()
        new_pts[idx] = np.clip(new_pts[idx] + move, -0.5, 0.5)
        
        new_r = _compute_radius(new_pts)
        if new_r > current_r:
            current_pts = new_pts
            current_r = new_r
    
    return current_pts, current_r

def construct():
    """Construct the optimal packing of 200 circles."""
    start_time = time.time()
    time_limit = 110  # Leave some margin
    
    best_pts = None
    best_r = 0.0
    
    # Strategy 1: Try different grid configurations with jitter
    for m in [13, 14, 15, 16]:
        if time.time() - start_time > time_limit:
            break
        for jitter in [0.0, 0.01, 0.02]:
            if time.time() - start_time > time_limit:
                break
            if jitter == 0.0:
                pts = _grid_initialization(m)
            else:
                pts = _jittered_grid_initialization(m, jitter)
            if len(pts) != N:
                continue
            r = _compute_radius(pts)
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
    
    # Strategy 2: Repulsion search from best grid
    if best_pts is not None:
        if time.time() - start_time < time_limit:
            opt_pts, opt_r = _repulsion_search(best_pts, max_iter=5000)
            if opt_r > best_r:
                best_r = opt_r
                best_pts = opt_pts
    
    # Strategy 3: Multiple random restarts with repulsion search
    n_restarts = 5
    for i in range(n_restarts):
        if time.time() - start_time > time_limit - 30:
            break
        
        # Random initialization
        pts = _random_initialization()
        
        # Quick repulsion search
        if time.time() - start_time < time_limit - 20:
            opt_pts, opt_r = _repulsion_search(pts, max_iter=3000)
            if opt_r > best_r:
                best_r = opt_r
                best_pts = opt_pts
    
    # Strategy 4: Hill climbing from best solution
    if best_pts is not None and time.time() - start_time < time_limit - 10:
        hc_pts, hc_r = _hill_climbing(best_pts, n_iterations=200, step_size=0.003)
        if hc_r > best_r:
            best_r = hc_r
            best_pts = hc_pts
    
    # Final repulsion search if time permits
    if best_pts is not None and time.time() - start_time < time_limit - 5:
        final_pts, final_r = _repulsion_search(best_pts, max_iter=5000)
        if final_r > best_r:
            best_r = final_r
            best_pts = final_pts
    
    # Ensure valid output
    if best_pts is None:
        # Fallback: simple grid
        m = 14
        pts = _grid_initialization(m)
        best_pts = pts
    
    # Final validation and clipping
    best_pts = np.clip(best_pts, -0.5, 0.5)
    
    return best_pts

# EVOLVE-BLOCK-END
