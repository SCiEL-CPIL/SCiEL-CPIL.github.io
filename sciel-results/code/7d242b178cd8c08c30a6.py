import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_violation(points, R):
    """
    Compute the sum of squared violations for a given configuration and target radius R.
    Violations:
    1. Circle-circle overlap: max(0, 2R - dist_ij)^2
    2. Circle-boundary overlap: max(0, R - dist_to_wall)^2
    """
    if not np.all(np.isfinite(points)):
        return np.inf
        
    # Ensure points are within bounds for calculation stability, though penalty handles it
    # We don't clip points in the objective, we let the penalty push them in.
    
    # 1. Pairwise distances
    # Use broadcasting for distance matrix
    # d[i, j] = ||p_i - p_j||
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Overlap penalty: if dist < 2R, penalty is (2R - dist)^2
    # We only care about i < j
    triu_idx = np.triu_indices(N, k=1)
    pair_dists = dists[triu_idx]
    overlaps = np.maximum(0.0, 2.0 * R - pair_dists)
    circle_penalty = np.sum(overlaps ** 2)
    
    # 2. Boundary penalties
    # Walls at x = -0.5, 0.5 and y = -0.5, 0.5
    # Distance to left wall: points[:, 0] - (-0.5) = points[:, 0] + 0.5
    # Distance to right wall: 0.5 - points[:, 0]
    # Distance to bottom wall: points[:, 1] + 0.5
    # Distance to top wall: 0.5 - points[:, 1]
    
    dx_left = points[:, 0] + 0.5
    dx_right = 0.5 - points[:, 0]
    dy_bottom = points[:, 1] + 0.5
    dy_top = 0.5 - points[:, 1]
    
    # Overlap if distance < R
    wall_overlaps = np.concatenate([
        np.maximum(0.0, R - dx_left),
        np.maximum(0.0, R - dx_right),
        np.maximum(0.0, R - dy_bottom),
        np.maximum(0.0, R - dy_top)
    ])
    
    wall_penalty = np.sum(wall_overlaps ** 2)
    
    return circle_penalty + wall_penalty

def _get_radius(points):
    """
    Calculate the maximum valid radius for the given points.
    """
    if not np.all(np.isfinite(points)):
        return -np.inf
        
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Wall distances
    x_dists = 0.5 - np.abs(points[:, 0])
    y_dists = 0.5 - np.abs(points[:, 1])
    min_wall_dist = np.min(np.concatenate([x_dists, y_dists]))
    
    radius = min(min_pair_dist / 2.0, min_wall_dist)
    return radius

def _generate_initializations():
    """
    Generate diverse initial configurations using a grid-based sampling strategy.
    """
    inits = []
    
    # 1. Hexagonal Lattice (Centered)
    # Create a hex grid
    a = 0.15 # spacing
    v1 = np.array([a, 0])
    v2 = np.array([a/2, a*np.sqrt(3)/2])
    pts = []
    for i in range(-10, 11):
        for j in range(-10, 11):
            p = i * v1 + j * v2
            if abs(p[0]) < 0.45 and abs(p[1]) < 0.45:
                pts.append(p)
    pts = np.array(pts)
    # If we have more than N, pick N closest to center or random subset
    if len(pts) >= N:
        dists = np.linalg.norm(pts, axis=1)
        indices = np.argsort(dists)
        inits.append(pts[indices[:N]].copy())
    else:
        # Pad with random if needed (unlikely for this grid size)
        inits.append(np.random.uniform(-0.4, 0.4, size=(N, 2)))

    # 2. Grid Sampling (Discrete-like)
    # Create a fine grid
    grid_size = 20
    xs = np.linspace(-0.4, 0.4, grid_size)
    ys = np.linspace(-0.4, 0.4, grid_size)
    grid_points = np.array(np.meshgrid(xs, ys)).T.reshape(-1, 2)
    
    # Randomly select N points from the grid multiple times
    for _ in range(10):
        if len(grid_points) >= N:
            indices = np.random.choice(len(grid_points), N, replace=False)
            inits.append(grid_points[indices].copy())
            
    # 3. Random Uniform
    for _ in range(5):
        inits.append(np.random.uniform(-0.4, 0.4, size=(N, 2)))
        
    return inits

def construct():
    start_time = time.time()
    deadline = 145.0 # Safety margin
    
    # Binary search parameters
    R_low = 0.0
    R_high = 0.075 # Slightly above known best 0.071377
    
    best_points = None
    best_radius = 0.0
    
    # Pre-generate initializations
    inits = _generate_initializations()
    
    # Tolerance for feasibility
    tol = 1e-6
    
    # Number of restarts per radius check
    num_restarts = 5
    
    # Binary search loop
    # We want to use as much time as possible.
    # Each iteration of binary search takes some time.
    # Let's run for a fixed number of steps or until time runs out.
    
    steps = 0
    max_steps = 25 # Approx 25 * (5 restarts * 1s) = 125s
    
    while time.time() < deadline - 10.0:
        R_mid = (R_low + R_high) / 2.0
        
        feasible = False
        best_violation_for_R = np.inf
        best_points_for_R = None
        
        # Try multiple initializations for this R
        # Shuffle inits to randomize order
        np.random.shuffle(inits)
        
        for init in inits[:num_restarts]:
            if time.time() > deadline - 10.0:
                break
                
            def objective(x_flat):
                pts = x_flat.reshape(N, 2)
                return _compute_violation(pts, R_mid)
            
            # Use Powell method, which is robust for non-smooth functions
            # maxiter should be small to allow many restarts
            res = minimize(objective, init.flatten(), method='Powell', 
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res.fun < best_violation_for_R:
                best_violation_for_R = res.fun
                best_points_for_R = res.x.reshape(N, 2)
                
            if best_violation_for_R < tol:
                feasible = True
                break
                
        if feasible:
            R_low = R_mid
            # Update best points if this radius is better than current best
            current_rad = _get_radius(best_points_for_R)
            if current_rad > best_radius:
                best_radius = current_rad
                best_points = best_points_for_R.copy()
        else:
            R_high = R_mid
            
        steps += 1
        
    # Final Polish: 
    # If we found a feasible R_low, try to optimize the radius directly from that point
    if best_points is not None and time.time() < deadline - 5.0:
        def neg_radius_obj(x_flat):
            pts = x_flat.reshape(N, 2)
            return -_get_radius(pts)
        
        # Use Nelder-Mead for final fine-tuning
        try:
            res = minimize(neg_radius_obj, best_points.flatten(), method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
            final_rad = -res.fun
            if final_rad > best_radius:
                best_radius = final_rad
                best_points = res.x.reshape(N, 2)
        except Exception:
            pass
            
    # Ensure validity and clipping
    if best_points is None:
        # Fallback: return a valid random configuration
        best_points = np.random.uniform(-0.4, 0.4, size=(N, 2))
        
    best_points = np.clip(best_points, -0.5, 0.5)
    
    return best_points
