import numpy as np
from scipy.optimize import minimize

N = 100

import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
def _local_search(centers, steps=1000, alpha_start=0.02, alpha_end=0.0005):
    """
    Perform iterative local search to maximize the minimum distance.
    centers: (N, 3) array
    steps: number of full passes over all particles
    alpha_start, alpha_end: step size decay
    """
    n = len(centers)
    centers = centers.copy()
    
    # Decay schedule
    alphas = np.geomspace(alpha_start, alpha_end, steps)
    
    for step in range(steps):
        alpha = alphas[step]
        # Shuffle indices to avoid bias
        perm = np.random.permutation(n)
        for i in perm:
            c_i = centers[i]
            
            # Calculate distances to all other points
            diff = centers - c_i
            dists = np.linalg.norm(diff, axis=1)
            dists[i] = np.inf # Ignore self
            
            # Find nearest neighbor
            j = np.argmin(dists)
            d_nn = dists[j]
            
            # Calculate boundary distances
            b_x = 0.5 - np.abs(c_i[0])
            b_y = 0.5 - np.abs(c_i[1])
            b_z = 0.5 - np.abs(c_i[2])
            
            # Determine the limiting constraint
            if d_nn < b_x:
                # Nearest constraint is neighbor j
                if d_nn > 1e-10:
                    grad = (c_i - centers[j]) / d_nn
                else:
                    grad = np.random.randn(3)
            elif b_x < b_y and b_x < b_z:
                grad = np.zeros(3)
                if c_i[0] > 0:
                    grad[0] = -1.0
                else:
                    grad[0] = 1.0
            elif b_y < b_x and b_y < b_z:
                grad = np.zeros(3)
                if c_i[1] > 0:
                    grad[1] = -1.0
                else:
                    grad[1] = 1.0
            else:
                grad = np.zeros(3)
                if c_i[2] > 0:
                    grad[2] = -1.0
                else:
                    grad[2] = 1.0
            
            # Update position
            c_i_new = c_i + alpha * grad
            c_i_new = np.clip(c_i_new, -0.5, 0.5)
            centers[i] = c_i_new
            
    return centers

def _evaluate(centers):
    """
    Calculate the radius for a given configuration.
    """
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    dx = 0.5 - np.abs(centers[:, 0])
    dy = 0.5 - np.abs(centers[:, 1])
    dz = 0.5 - np.abs(centers[:, 2])
    min_bound_dist = np.min(np.minimum(np.minimum(dx, dy), dz))
    
    r = 0.5 * min(min_pair_dist, min_bound_dist)
    return r

def construct():
    best_radius = 0.0
    best_centers = None
    
    start_time = time.time()
    time_limit = 110.0
    
    n_restarts = 15
    steps_per_restart = 1500
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
            
        rng = np.random.default_rng(42 + restart)
        centers = rng.uniform(-0.5, 0.5, size=(N, 3))
        
        centers = _local_search(centers, steps=steps_per_restart, alpha_start=0.02, alpha_end=0.0005)
        
        r = _evaluate(centers)
        
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
            
    if best_centers is None:
        m = int(np.ceil(N ** (1.0 / 3.0)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers
# EVOLVE-BLOCK-END
