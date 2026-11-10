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

def _smooth_objective(x):
    """
    Smooth objective function to minimize.
    Minimizes the sum of inverse squared distances to encourage separation.
    """
    pts = x.reshape(-1, 2)
    
    # Pairwise distances
    # Using broadcasting for N=50 is fast enough (50x50 matrix)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Set diagonal to inf to ignore self-distance
    np.fill_diagonal(dist_sq, np.inf)
    
    # Inverse squared distances (add epsilon to avoid division by zero)
    epsilon = 1e-12
    inv_dist_sq = 1.0 / (dist_sq + epsilon)
    
    # Sum of pairwise repulsions (only upper triangle to avoid double counting, or just sum all / 2)
    pair_repulsion = np.sum(inv_dist_sq) / 2.0
    
    # Boundary repulsions
    # Distance to nearest wall
    dx = 0.5 - np.abs(pts[:, 0])
    dy = 0.5 - np.abs(pts[:, 1])
    wall_dist = np.minimum(dx, dy)
    
    # Ensure wall_dist is not negative (if point is outside, penalty should be huge)
    # But we use bounds, so points should stay inside.
    # If a point goes slightly out due to numerical errors, wall_dist < 0.
    # We want to penalize being outside strongly.
    # Using 1/wall_dist^2 is problematic if wall_dist is 0 or negative.
    # Let's use a penalty that is smooth and large when outside.
    # If wall_dist <= 0, return a large value.
    if np.any(wall_dist <= 0):
        return 1e6
    
    wall_repulsion = np.sum(1.0 / (wall_dist + epsilon)**2)
    
    return pair_repulsion + wall_repulsion

def _initial_grid_7x7_plus_1():
    """Create a 7x7 grid (49 points) and add one point at the center."""
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    pts = np.array([(x, y) for y in ticks for x in ticks])
    # Add center point
    pts = np.vstack([pts, [[0.0, 0.0]]])
    return pts[:N].astype(np.float64)

def _initial_random(rng):
    """Create a random initial configuration."""
    return rng.uniform(-0.5, 0.5, size=(N, 2))

def construct():
    """Construct centers of N equal circles inside the square [-1/2, 1/2]^2."""
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    best_pts = None
    best_radius = 0.0
    
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    rng = np.random.default_rng(42)
    
    # Initializations to try
    inits = []
    
    # 1. Grid + Center
    inits.append(_initial_grid_7x7_plus_1())
    
    # 2. Random
    inits.append(_initial_random(rng))
    
    # 3. Perturbed Grid
    grid = _initial_grid_7x7_plus_1()
    inits.append(grid + rng.normal(0, 0.05, size=grid.shape))
    
    for i, init in enumerate(inits):
        if time.time() - start_time > time_limit - 20:
            break
            
        x0 = init.flatten().astype(np.float64)
        
        try:
            # Use L-BFGS-B for faster convergence on smooth problems
            res = minimize(
                _smooth_objective, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < 1e5: # Check if it didn't fail catastrophically
                pts = res.x.reshape(-1, 2)
                # Clip to ensure validity
                pts = np.clip(pts, -0.5, 0.5)
                
                r = _compute_radius(pts)
                
                if r > best_radius:
                    best_radius = r
                    best_pts = pts.copy()
                    
        except Exception:
            continue
            
        # Check time
        if time.time() - start_time > time_limit - 10:
            break
        
        # If we found a very good radius close to theoretical max, we might stop early
        # Theoretical max for 50 circles is roughly 0.07137
        if best_radius > 0.0713:
             # Do one more refinement if time permits
             if time.time() - start_time < time_limit - 30:
                 try:
                     res2 = minimize(
                         _smooth_objective, 
                         best_pts.flatten(), 
                         method='L-BFGS-B', 
                         bounds=bounds,
                         options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
                     )
                     pts2 = res2.x.reshape(-1, 2)
                     pts2 = np.clip(pts2, -0.5, 0.5)
                     r2 = _compute_radius(pts2)
                     if r2 > best_radius:
                         best_radius = r2
                         best_pts = pts2.copy()
                 except Exception:
                     pass
             break

    # Fallback: if optimization failed or returned bad results, use the best initial or a safe grid
    if best_pts is None or best_radius < 0.05:
        # Use the simple grid as fallback, ensuring it's valid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid_pts = np.array([(x, y) for y in ticks for x in ticks])
        best_pts = grid_pts[:N].astype(np.float64)
        best_radius = _compute_radius(best_pts)
        
    return best_pts

# EVOLVE-BLOCK-END
