"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

START_TIME = time.time()
TIME_LIMIT = 140.0  # Leave 10s safety margin before 150s kill
SAFETY_MARGIN = 10.0


def _remaining_time():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)


def _compute_radius(centers):
    """Compute the maximum radius for given centers."""
    if len(centers) == 0:
        return 0.0
    
    # Distance to walls: min_i min_k (0.5 - |c_ik|)
    wall_dist = np.min(0.5 - np.abs(centers), axis=None)
    
    # Pairwise distances
    n = len(centers)
    if n < 2:
        return float(wall_dist)
    
    # Compute all pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(float(wall_dist), float(min_pair_dist / 2.0))


def _initial_grid():
    """Create initial grid arrangement."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _optimize_positions(centers):
    """Optimize positions to maximize minimum radius."""
    def neg_radius(x):
        """Negative radius for minimization."""
        c = x.reshape(-1, 2)
        return -_compute_radius(c)
    
    # Use Nelder-Mead which doesn't require gradients
    result = minimize(neg_radius, centers.flatten(), method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
    return result.x.reshape(-1, 2)


def construct():
    """Construct optimal circle packing."""
    # Start with grid
    centers = _initial_grid()
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    # Try optimization if we have time
    if _remaining_time() > 5.0:
        try:
            optimized = _optimize_positions(centers)
            opt_radius = _compute_radius(optimized)
            
            if opt_radius > best_radius:
                best_radius = opt_radius
                best_centers = optimized
        except Exception:
            pass
    
    # Ensure all centers are within bounds
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
# EVOLVE-BLOCK-END
