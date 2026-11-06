"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
def _compute_radius(centers):
    """Compute the largest common radius for given centers in [-0.5, 0.5]^3."""
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Wall distance
    wall = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall = float(np.min(wall))
    if min_wall < 0.0:
        return 0.0
    # Pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = float(np.min(dists)) / 2.0
    return min(min_wall, min_pair)

def _penalty_objective(x):
    """Objective function for scipy minimize: minimize negative soft-min radius."""
    centers = x.reshape(N, 3)
    centers = np.clip(centers, -0.49, 0.49)
    
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2) + 1e-12)
    np.fill_diagonal(dists, np.inf)
    
    walls = 0.5 - np.max(np.abs(centers), axis=1)
    
    iu = np.triu_indices(N, k=1)
    pair_terms = dists[iu] / 2.0
    all_terms = np.concatenate([pair_terms, walls])
    
    t = 0.01
    m = np.min(all_terms)
    shifted = -all_terms / t
    shifted -= np.max(shifted)
    s = np.sum(np.exp(shifted))
    soft_min_val = m + t * np.log(s)
    
    return -soft_min_val

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initial configuration: 5x5x4 grid
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.4, 0.4, nz)
    centers = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    centers = centers[:N]
    
    best_centers = centers.copy()
    
    x0 = centers.flatten()
    bounds = [(-0.49, 0.49)] * (N * 3)
    
    try:
        res = minimize(_penalty_objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-10})
        if time.time() < deadline:
            best_centers = res.x.reshape(N, 3)
    except Exception:
        pass
        
    best_centers = np.clip(best_centers, -0.49, 0.49)
    return best_centers
# EVOLVE-BLOCK-END
