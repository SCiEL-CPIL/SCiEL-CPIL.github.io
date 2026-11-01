import numpy as np
from scipy.optimize import minimize

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
    # We need min over j != i of ||p_i - p_j|| / 2
    # Compute pairwise distances
    # diff = p[:, np.newaxis, :] - p[np.newaxis, :, :] # (N, N, 2)
    # dists = np.linalg.norm(diff, axis=2) # (N, N)
    # np.fill_diagonal(dists, np.inf)
    # min_neighbor_dist = np.min(dists, axis=1) / 2.0
    
    # More efficient vectorized calculation
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_neighbor_dist = np.min(dists, axis=1) / 2.0
    
    radii = np.minimum(min_wall_dist, min_neighbor_dist)
    return np.sum(radii)

def construct():
    # Initial grid
    side = int(np.ceil(np.sqrt(N)))
    xs = (np.arange(side) + 0.5) / side
    ys = (np.arange(side) + 0.5) / side
    init_points = []
    for i in range(side):
        for j in range(side):
            if len(init_points) < N:
                init_points.append([xs[i], ys[j]])
    init_c = np.array(init_points).flatten()
    
    # Optimize
    # Use Powell or Nelder-Mead
    # Powell is robust for non-smooth
    res = minimize(lambda c: -sum_radii(c), init_c, method='Powell', options={'maxiter': 1000, 'disp': False})
    
    best_c = res.x
    best_val = -res.fun
    
    # Try a random restart if time allows? 
    # For now, just one good optimization run.
    # To improve, we can run a few local searches from slightly perturbed grids.
    
    # Let's try 3 different initializations: Grid, Grid+noise, Random
    best_c = init_c
    best_val = sum_radii(init_c)
    
    for _ in range(3):
        # Perturb grid
        noise = np.random.normal(0, 0.05, size=init_c.shape)
        c_init = np.clip(init_c + noise, 0.01, 0.99)
        res = minimize(lambda c: -sum_radii(c), c_init, method='Powell', options={'maxiter': 500, 'disp': False})
        val = -res.fun
        if val > best_val:
            best_val = val
            best_c = res.x
            
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
    
    result = np.column_stack((p, radii))
    return result
