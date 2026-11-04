import numpy as np
from scipy.optimize import minimize

N = 100

def objective(c):
    """
    Objective function to minimize.
    We want to maximize r, so we minimize -r.
    r = min(min_pair_dist / 2, min_boundary_dist)
    """
    c = c.reshape(N, 3)
    
    # Calculate pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Set diagonal to infinity to ignore self-distance
    np.fill_diagonal(dists, np.inf)
    
    min_pair = np.min(dists)
    
    # Calculate boundary distances
    max_abs_coord = np.max(np.abs(c), axis=1)
    min_boundary = 0.5 - np.max(max_abs_coord)
    
    r = min(min_pair / 2.0, min_boundary)
    
    return -r

def construct():
    # Multiple starting points
    starts = []
    
    # 1. Jittered Grid
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.375, 0.375, nz)
    
    grid_centers = np.array([(x, y, z) for x in xs for y in ys for z in zs])
    jitter = np.random.normal(0, 0.01, grid_centers.shape)
    grid_centers = np.clip(grid_centers + jitter, -0.45, 0.45)
    starts.append(grid_centers.flatten())
    
    # 2. Random Uniform Starts
    for _ in range(5):
        random_centers = np.random.uniform(-0.4, 0.4, (N, 3))
        starts.append(random_centers.flatten())
    
    best_center = None
    best_score = -np.inf
    
    # Run Powell optimization for each start
    for start in starts:
        try:
            res = minimize(objective, start, method='Powell', options={'maxiter': 2000, 'maxfev': 20000})
            current_r = -res.fun
            if current_r > best_score:
                best_score = current_r
                best_center = res.x.reshape(N, 3)
        except Exception:
            pass
            
    if best_center is None:
        best_center = grid_centers
        
    return best_center
