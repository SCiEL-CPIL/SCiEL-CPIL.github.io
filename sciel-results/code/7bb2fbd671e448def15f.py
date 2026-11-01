import numpy as np
import scipy.optimize
import time

N = 50

def compute_radius(centers):
    """
    Computes the maximum common radius for a set of centers.
    centers: numpy array of shape (N, 3)
    """
    # Ensure centers is a numpy array
    centers = np.asarray(centers)
    
    # Calculate pairwise distances
    # Use broadcasting
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    
    # Calculate distance to walls
    # Cube is [-0.5, 0.5]
    # Distance to nearest wall for a point (x,y,z) is min(0.5-x, 0.5+x, 0.5-y, 0.5+y, 0.5-z, 0.5+z)
    # Which is 0.5 - max(|x|, |y|, |z|) ? No.
    # Distance to x=0.5 is 0.5-x. Distance to x=-0.5 is 0.5+x.
    # Min distance in x direction is 0.5 - |x|.
    # Min distance to any wall is min(0.5-|x|, 0.5-|y|, 0.5-|z|).
    # Wait, the sphere must be inside.
    # The radius is limited by the distance to the closest boundary.
    # The closest boundary to a point (x,y,z) is min(0.5-x, 0.5+x, 0.5-y, 0.5+y, 0.5-z, 0.5+z).
    # This is equivalent to 0.5 - max(|x|, |y|, |z|).
    # Let's verify: if x=0.4, y=0, z=0. max|.| = 0.4. 0.5-0.4 = 0.1.
    # Dist to x=0.5 is 0.1. Dist to x=-0.5 is 0.9. Min is 0.1. Correct.
    
    max_abs_coord = np.max(np.abs(centers), axis=1)
    min_wall_dist = np.min(0.5 - max_abs_coord)
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def objective(x):
    centers = x.reshape(N, 3)
    r = compute_radius(centers)
    return -r

def construct():
    start_time = time.time()
    deadline = 110  # Leave 10s buffer
    
    best_centers = None
    best_score = -np.inf
    
    # Strategy 1: 4x4x4 Grid
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # We have 64 points, need 50.
    # Randomly select 50 to break symmetry and allow better packing
    # But for the first try, let's just take the first 50 and perturb.
    np.random.seed(42)
    
    # Try multiple random subsets of the grid to break symmetry
    for attempt in range(10):
        if time.time() - start_time > deadline:
            break
            
        # Select 50 indices from the 64 grid points
        indices = np.random.choice(len(grid), N, replace=False)
        initial_centers = grid[indices].copy()
        
        # Add small noise
        noise_scale = 0.05
        initial_centers += np.random.normal(0, noise_scale, initial_centers.shape)
        
        x0 = initial_centers.flatten()
        
        try:
            # Use Powell method
            # maxiter and maxfev control the runtime
            # We don't know how many fev per second, so we use a generous limit and check time
            res = scipy.optimize.minimize(objective, x0, method='Powell', 
                                           options={'maxiter': 200, 'maxfev': 2000})
            
            if res.x is not None:
                candidate = res.x.reshape(N, 3)
                score = compute_radius(candidate)
                
                if score > best_score:
                    best_score = score
                    best_centers = candidate
                    
        except Exception:
            continue

    # Strategy 2: 3x4x5 Grid (60 points)
    if time.time() - start_time < deadline - 10:
        m_x, m_y, m_z = 3, 4, 5
        ticks_x = (np.arange(m_x) + 0.5) / m_x - 0.5
        ticks_y = (np.arange(m_y) + 0.5) / m_y - 0.5
        ticks_z = (np.arange(m_z) + 0.5) / m_z - 0.5
        grid2 = np.array([(x, y, z) for z in ticks_z for y in ticks_y for x in ticks_x])
        
        for attempt in range(5):
            if time.time() - start_time > deadline:
                break
                
            indices = np.random.choice(len(grid2), N, replace=False)
            initial_centers = grid2[indices].copy()
            initial_centers += np.random.normal(0, 0.05, initial_centers.shape)
            
            x0 = initial_centers.flatten()
            
            try:
                res = scipy.optimize.minimize(objective, x0, method='Powell', 
                                               options={'maxiter': 200, 'maxfev': 2000})
                if res.x is not None:
                    candidate = res.x.reshape(N, 3)
                    score = compute_radius(candidate)
                    
                    if score > best_score:
                        best_score = score
                        best_centers = candidate
            except Exception:
                continue

    # Fallback: Simple Grid if optimization failed or returned garbage
    if best_centers is None or not np.all(np.isfinite(best_centers)):
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]

    return best_centers
