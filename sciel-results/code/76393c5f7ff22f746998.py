"""Rewritten program for packing 200 equal circles in a unit square."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import time
import random

N = 200

# EVOLVE-BLOCK-START

def _compute_radius_fast(pts):
    """
    Compute the maximum radius for given centers using KDTree for speed.
    Returns the minimum of (min_pairwise_dist / 2) and (min_boundary_dist).
    """
    if not np.all(np.isfinite(pts)):
        return 0.0
    
    # Boundary distance
    # Distance to walls for each point
    dist_x = np.minimum(0.5 - pts[:, 0], pts[:, 0] + 0.5)
    dist_y = np.minimum(0.5 - pts[:, 1], pts[:, 1] + 0.5)
    min_bound = np.min(np.minimum(dist_x, dist_y))
    
    # Pairwise distance using KDTree
    tree = cKDTree(pts)
    # Query 2 nearest neighbors for each point (one is self, one is neighbor)
    # Actually, we want the closest *other* point.
    # query k=2 returns self and nearest neighbor.
    try:
        d, _ = tree.query(pts, k=2)
        min_pair = np.min(d[:, 1])
    except Exception:
        min_pair = 0.0
        
    return min(min_pair / 2.0, min_bound)

def _hex_initialization():
    """
    Create a hexagonal grid initialization.
    Hexagonal packing is denser than square.
    """
    # Generate a hex grid covering the square
    # Side length s. Height h = s * sqrt(3)/2.
    # We estimate s to fit roughly N points.
    # Area of hex cell ~ 0.433 * s^2.
    # Total area 1. N=200. Area per point 0.005.
    # s^2 ~ 0.005 / 0.433 ~ 0.0115. s ~ 0.1.
    
    s = 0.09 
    h = s * np.sqrt(3) / 2
    
    rows = int(1.0 / h) + 2
    cols = int(1.0 / s) + 2
    
    pts = []
    for i in range(rows):
        y = -0.5 + (i + 0.5) * h
        if y > 0.5: break
        offset = 0.0 if i % 2 == 0 else s / 2.0
        for j in range(cols):
            x = -0.5 + (j + 0.5) * s + offset
            if x > 0.5: break
            pts.append([x, y])
            
    pts = np.array(pts)
    if len(pts) < N:
        # If we didn't get enough, try a smaller spacing
        s = 0.08
        h = s * np.sqrt(3) / 2
        pts = []
        rows = int(1.0 / h) + 2
        cols = int(1.0 / s) + 2
        for i in range(rows):
            y = -0.5 + (i + 0.5) * h
            if y > 0.5: break
            offset = 0.0 if i % 2 == 0 else s / 2.0
            for j in range(cols):
                x = -0.5 + (j + 0.5) * s + offset
                if x > 0.5: break
                pts.append([x, y])
        pts = np.array(pts)
        
    if len(pts) >= N:
        # Select N points. Taking the first N might be biased, but SA will fix it.
        # To be safer, we can select points uniformly from the generated grid.
        indices = np.random.choice(len(pts), N, replace=False)
        return pts[indices]
    else:
        # Fallback to random if hex grid failed
        return np.random.uniform(-0.5, 0.5, size=(N, 2))

def _simulated_annealing(pts, time_limit=10.0):
    """
    Perform Simulated Annealing to maximize the minimum radius.
    """
    start_time = time.time()
    current_pts = pts.copy()
    current_r = _compute_radius_fast(current_pts)
    
    # Initial temperature
    T = 0.01
    T_min = 1e-6
    cooling_rate = 0.999
    
    # Move size
    move_sigma = 0.01
    
    best_pts = current_pts.copy()
    best_r = current_r
    
    while time.time() - start_time < time_limit:
        # Pick a random point to move
        idx = np.random.randint(0, N)
        
        # Generate a random move
        move = np.random.normal(0, move_sigma, size=2)
        new_pts = current_pts.copy()
        new_pts[idx] = current_pts[idx] + move
        
        # Clamp to box
        new_pts[idx] = np.clip(new_pts[idx], -0.5, 0.5)
        
        new_r = _compute_radius_fast(new_pts)
        
        # Calculate delta
        delta = new_r - current_r
        
        # Acceptance criterion
        if delta > 0:
            # Improvement
            current_pts = new_pts
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_pts = current_pts.copy()
        else:
            # Accept with probability exp(delta / T)
            prob = np.exp(delta / T)
            if np.random.random() < prob:
                current_pts = new_pts
                current_r = new_r
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
        
    return best_pts, best_r

def construct():
    """Construct the optimal packing of 200 circles."""
    start_time = time.time()
    time_limit = 110  # Leave some margin
    
    best_pts = None
    best_r = 0.0
    
    # Strategy 1: Hexagonal Grid Initialization + SA
    # Run multiple SA restarts from slightly perturbed hex grids
    n_restarts = 5
    time_per_restart = (time_limit - 10) / n_restarts
    
    for i in range(n_restarts):
        if time.time() - start_time > time_limit - 5:
            break
            
        # Get hex grid
        pts = _hex_initialization()
        
        # Add small random jitter to break symmetry
        pts = pts + np.random.normal(0, 0.005, size=pts.shape)
        pts = np.clip(pts, -0.5, 0.5)
        
        # Run SA
        sa_time = time_per_restart
        opt_pts, opt_r = _simulated_annealing(pts, time_limit=sa_time)
        
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts
    
    # Strategy 2: If time remains, do a final polish with L-BFGS-B on the best solution
    if best_pts is not None and time.time() - start_time < time_limit - 5:
        # L-BFGS-B is good for fine-tuning
        # We need an objective function that L-BFGS-B can use.
        # Since _compute_radius_fast is non-smooth, L-BFGS-B might struggle,
        # but it often works if initialized close to the optimum.
        # We'll use a simple objective: -radius
        def obj(x):
            p = x.reshape(-1, 2)
            return -_compute_radius_fast(p)
            
        x0 = best_pts.flatten()
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        try:
            result = minimize(
                obj, x0, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            if result.success or result.status == 1:
                final_pts = result.x.reshape(-1, 2)
                final_pts = np.clip(final_pts, -0.5, 0.5)
                final_r = _compute_radius_fast(final_pts)
                if final_r > best_r:
                    best_r = final_r
                    best_pts = final_pts
        except Exception:
            pass

    # Ensure valid output
    if best_pts is None:
        # Fallback
        best_pts = _hex_initialization()
        
    # Final validation and clipping
    best_pts = np.clip(best_pts, -0.5, 0.5)
    
    return best_pts

# EVOLVE-BLOCK-END
