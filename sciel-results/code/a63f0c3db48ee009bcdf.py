import numpy as np
from scipy.optimize import minimize
import time

N = 32

def sum_radii(c):
    """
    c: flat array of length 2*N, representing N centers (x, y)
    Returns: sum of inradii of the Voronoi cells clipped to [0,1]^2
    """
    p = c.reshape(N, 2)
    # Distances to walls
    dist_walls = np.minimum(p, 1 - p) # shape (N, 2)
    min_wall_dist = np.min(dist_walls, axis=1) # shape (N,)
    
    # Distances to other centers / 2
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_neighbor_dist = np.min(dists, axis=1) / 2.0
    
    radii = np.minimum(min_wall_dist, min_neighbor_dist)
    return np.sum(radii)

def create_grid_init(rows, cols, n):
    """Create a grid initialization with specified rows and cols, taking first n points"""
    xs = (np.arange(cols) + 0.5) / cols
    ys = (np.arange(rows) + 0.5) / rows
    points = []
    for i in range(rows):
        for j in range(cols):
            if len(points) < n:
                points.append([xs[j], ys[i]])
    return np.array(points).flatten()

def construct():
    start_time = time.time()
    timeout = 115  # Leave 5s margin
    
    best_c = None
    best_val = -np.inf
    
    # Generate a diverse set of initializations
    inits = []
    
    # 1. Regular Grids (Ensure rows*cols >= N)
    grid_configs = [(6, 6), (5, 7), (7, 5), (4, 8), (8, 4), (6, 5), (5, 6)]
    for r, c in grid_configs:
        inits.append(create_grid_init(r, c, N))
        
    # 2. Random Inits
    for i in range(10):
        inits.append(np.random.uniform(0.1, 0.9, size=2*N))
        
    # 3. Perturbed Grids (from the first 3 grid configs)
    for base in inits[:3]:
        for scale in [0.05, 0.1, 0.2]:
            noise = np.random.normal(0, scale, size=2*N)
            perturbed = np.clip(base + noise, 0.01, 0.99)
            inits.append(perturbed)

    # Optimization Loop
    candidates = []
    
    for idx, c_init in enumerate(inits):
        if time.time() - start_time > timeout - 10:
            break
            
        # Try Powell
        try:
            res = minimize(lambda c: -sum_radii(c), c_init, method='Powell', 
                          options={'maxiter': 500, 'disp': False, 'xtol': 1e-8, 'ftol': 1e-8})
            
            val = -res.fun
            if val > best_val:
                best_val = val
                best_c = res.x.copy()
            
            # Keep top candidates for final refinement
            candidates.append((val, res.x.copy()))
            
        except Exception:
            pass

    # Sort candidates by value
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Take the top 5 candidates and refine them further
    # This helps escape local minima if the first pass didn't converge fully
    for val, c_cand in candidates[:5]:
        if time.time() - start_time > timeout - 5:
            break
            
        try:
            # More iterations for refinement
            res = minimize(lambda c: -sum_radii(c), c_cand, method='Powell', 
                          options={'maxiter': 1000, 'disp': False, 'xtol': 1e-10, 'ftol': 1e-10})
            
            val = -res.fun
            if val > best_val:
                best_val = val
                best_c = res.x.copy()
                
        except Exception:
            pass

    # If no valid result, fall back to grid
    if best_c is None:
        best_c = create_grid_init(6, 6, N)
        best_val = sum_radii(best_c)
    
    # Convert to (N, 3) array
    p = best_c.reshape(N, 2)
    # Recalculate radii for the final centers
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_neighbor_dist = np.min(dists, axis=1) / 2.0
    dist_walls = np.minimum(p, 1 - p)
    min_wall_dist = np.min(dist_walls, axis=1)
    radii = np.minimum(min_wall_dist, min_neighbor_dist)
    
    # Shrink radii slightly to account for floating point precision in overlap checks
    radii = radii * 0.999
    
    result = np.column_stack((p, radii))
    return result
