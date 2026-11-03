import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time

N = 200
_TIME_LIMIT = 145  # Leave 5s margin for 150s kill

def _score_from_points(pts):
    """Calculate the maximum radius for a set of points."""
    if len(pts) < 2:
        return 0.0
    
    # Check bounds
    if np.any(pts[:, 0] < -0.5) or np.any(pts[:, 0] > 0.5) or \
       np.any(pts[:, 1] < -0.5) or np.any(pts[:, 1] > 0.5):
        return 0.0

    # Min distance to wall
    wall_dist = np.min([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1]),
    ])
    
    # Min pairwise distance
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D)
    
    r = min(wall_dist, min_pair / 2.0)
    return r if r > 0 else 0.0

def _generate_hex_grid(resolution):
    """
    Generate points on a hexagonal grid inside the square [-0.5, 0.5]^2.
    resolution controls the density.
    """
    # Hex grid spacing
    # For a hex grid, if we want 'res' points across, spacing is 1/res
    # But hex grids are denser. Let's just use a fine square grid and offset odd rows.
    # Actually, a true hex grid is better.
    
    # Let's generate a fine square grid first, then shift alternating rows.
    # To get a hex-like structure, the vertical spacing should be dx * sqrt(3)/2.
    # Let dx = 1.0 / res.
    # Then dy = dx * sqrt(3) / 2.
    
    dx = 1.0 / resolution
    dy = dx * np.sqrt(3) / 2.0
    
    points = []
    
    # Number of rows
    num_rows = int(1.0 / dy) + 2
    num_cols = int(1.0 / dx) + 2
    
    for i in range(num_rows):
        y = -0.5 + i * dy
        if y > 0.5: break
        # Offset for hex packing
        if i % 2 == 0:
            offset = 0.0
        else:
            offset = dx / 2.0
            
        for j in range(num_cols):
            x = -0.5 + j * dx + offset
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
                
    return np.array(points)

def _greedy_select(grid_points, n_points):
    """
    Select n_points from grid_points to maximize the minimum distance
    to other selected points and the walls.
    """
    selected = []
    remaining = list(range(len(grid_points)))
    
    # To speed up, we can precompute wall distances for all grid points
    # wall_dist[i] = min distance of grid_points[i] to any wall
    x, y = grid_points[:, 0], grid_points[:, 1]
    wall_dists = np.min([
        x + 0.5,
        0.5 - x,
        y + 0.5,
        0.5 - y
    ], axis=0)
    
    for _ in range(n_points):
        best_idx = -1
        best_score = -1.0
        
        # We want to maximize min(wall_dist, min_dist_to_selected)
        # For each candidate in remaining:
        # score = min(wall_dist[c], min_{s in selected} dist(c, s))
        
        # Optimization: if selected is empty, just pick max wall_dist
        if not selected:
            best_idx = remaining[np.argmax(wall_dists[remaining])]
        else:
            sel_pts = np.array(selected)
            # Compute distances from all remaining candidates to all selected points
            # This is O(R * S). R is remaining, S is selected.
            # We can do this in chunks or just vectorize if R*S is small.
            # With resolution 50, grid size ~ 2000. N=200.
            # 2000 * 200 = 400,000 ops per step. 200 steps = 80M ops. Acceptable.
            
            rem_pts = grid_points[remaining]
            D = cdist(rem_pts, sel_pts)
            min_d_to_sel = np.min(D, axis=1)
            
            # Effective radius for this candidate
            eff_r = np.min(wall_dists[remaining], min_d_to_sel)
            
            # Pick the one with max eff_r
            local_best = np.argmax(eff_r)
            best_idx = remaining[local_best]
            best_score = eff_r[local_best]
            
            # If best_score is very small, we might have issues, but greedy usually works.
            
        selected.append(grid_points[best_idx])
        remaining.remove(best_idx)
        
    return np.array(selected)

def _polish(pts, time_limit):
    """
    Refine points using gradient descent to maximize min distance.
    Objective: minimize sum(1/d_ij^2) + sum(1/w_i^2)
    """
    t0 = time.time()
    
    def objective(x):
        p = x.reshape(-1, 2)
        # Bounds penalty
        if np.any(p[:, 0] < -0.5) or np.any(p[:, 0] > 0.5) or \
           np.any(p[:, 1] < -0.5) or np.any(p[:, 1] > 0.5):
            return 1e10
        
        # Pairwise distances
        D = cdist(p, p)
        np.fill_diagonal(D, np.inf)
        
        # Wall distances
        wall_x_left = p[:, 0] + 0.5
        wall_x_right = 0.5 - p[:, 0]
        wall_y_left = p[:, 1] + 0.5
        wall_y_right = 0.5 - p[:, 1]
        
        # Avoid division by zero
        eps = 1e-8
        pair_term = np.sum(1.0 / (D + eps)**2)
        wall_term = np.sum(1.0 / (wall_x_left + eps)**2) + \
                    np.sum(1.0 / (wall_x_right + eps)**2) + \
                    np.sum(1.0 / (wall_y_left + eps)**2) + \
                    np.sum(1.0 / (wall_y_right + eps)**2)
        
        return pair_term + wall_term

    x0 = pts.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    try:
        # Use L-BFGS-B with a reasonable number of iterations
        # maxiter needs to be enough to converge.
        # We check time inside the callback if possible, but minimize doesn't support that easily.
        # We'll just set a high maxiter and rely on the time check afterwards, 
        # or use a callback to break early.
        
        def callback(xk):
            if time.time() - t0 > time_limit:
                raise KeyboardInterrupt
        
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8},
                       callback=callback)
        
        new_pts = res.x.reshape(-1, 2)
        
        r_old = _score_from_points(pts)
        r_new = _score_from_points(new_pts)
        
        if r_new > r_old:
            return new_pts
        else:
            return pts
            
    except KeyboardInterrupt:
        return pts
    except Exception:
        return pts

def construct():
    t0 = time.time()
    best_pts = None
    best_r = 0.0
    
    # 1. Generate a fine hex grid
    # Resolution 40-60 should be enough. 40 -> ~1600 points.
    grid_res = 50
    grid_points = _generate_hex_grid(grid_res)
    
    # 2. Greedy selection
    initial_pts = _greedy_select(grid_points, N)
    r_init = _score_from_points(initial_pts)
    
    if r_init > best_r:
        best_r = r_init
        best_pts = initial_pts
        
    # 3. Polish
    time_left = _TIME_LIMIT - (time.time() - t0)
    if time_left > 10:
        polished_pts = _polish(best_pts, time_left - 5)
        r_polished = _score_from_points(polished_pts)
        
        if r_polished > best_r:
            best_r = r_polished
            best_pts = polished_pts
            
    # 4. If time remains, try a different grid offset or resolution
    time_left = _TIME_LIMIT - (time.time() - t0)
    if time_left > 20:
        # Try a slightly different grid to escape local minima
        grid_res2 = 45
        grid_points2 = _generate_hex_grid(grid_res2)
        # Add a small random shift to break symmetry
        shift = np.random.uniform(-0.01, 0.01)
        grid_points2[:, 0] += shift
        
        initial_pts2 = _greedy_select(grid_points2, N)
        r_init2 = _score_from_points(initial_pts2)
        
        if r_init2 > best_r:
            best_r = r_init2
            best_pts = initial_pts2
            
        time_left = _TIME_LIMIT - (time.time() - t0)
        if time_left > 10:
            polished_pts2 = _polish(best_pts, time_left - 5)
            r_polished2 = _score_from_points(polished_pts2)
            
            if r_polished2 > best_r:
                best_r = r_polished2
                best_pts = polished_pts2

    # Fallback if something went wrong
    if best_pts is None:
        # Simple grid fallback
        rows, cols = 14, 15
        dx = 1.0 / cols
        dy = 1.0 / rows
        xs = -0.5 + (np.arange(cols) + 0.5) * dx
        ys = -0.5 + (np.arange(rows) + 0.5) * dy
        grid = np.array([(x, y) for y in ys for x in xs])
        best_pts = grid[:N]
        
    return best_pts
