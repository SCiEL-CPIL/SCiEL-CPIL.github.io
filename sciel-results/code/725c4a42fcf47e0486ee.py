"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

N = 50

def construct():
    """
    Place 50 equal circles in a unit circle to maximize the common radius.
    Uses random initialization and Nelder-Mead optimization with a smooth
    approximation of the minimum radius objective.
    """
    
    def get_radius(x):
        pts = x.reshape(N, 2)
        d_origin = np.linalg.norm(pts, axis=1)
        diff = pts[:, None, :] - pts[None, :, :]
        d2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(d2, np.inf)
        min_pair_dist = np.sqrt(np.min(d2))
        min_boundary_dist = 1 - np.max(d_origin)
        return min(min_pair_dist / 2, min_boundary_dist)

    def objective(x, alpha=100.0):
        """
        Negative of the soft-minimum of all pairwise half-distances and boundary distances.
        alpha controls the sharpness of the approximation.
        """
        pts = x.reshape(N, 2)
        d_origin = np.linalg.norm(pts, axis=1)
        diff = pts[:, None, :] - pts[None, :, :]
        d2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(d2, np.inf)
        
        # Distances to consider: half pairwise distances and boundary distances
        d_pairs = np.sqrt(d2) / 2.0
        d_boundary = 1.0 - d_origin
        
        # Flatten pairwise distances and boundary distances
        all_dists = np.concatenate([d_pairs.flatten(), d_boundary.flatten()])
        
        # Clip to avoid numerical issues
        all_dists = np.maximum(all_dists, 1e-12)
        
        # Soft min calculation
        scaled = -alpha * all_dists
        max_scaled = np.max(scaled)
        log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        soft_min = -log_sum / alpha
        
        return -soft_min

    def random_init(seed):
        """Generate random points in the unit disk."""
        rng = np.random.default_rng(seed)
        # Uniform distribution in disk
        r = np.sqrt(rng.uniform(0, 1, N))
        theta = rng.uniform(0, 2 * np.pi, N)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        pts = np.column_stack([x, y])
        return pts.flatten()

    best_x = None
    best_r = -1.0
    
    # Perform multiple random restarts
    num_restarts = 8
    
    for i in range(num_restarts):
        seed = 100 + i
        x0 = random_init(seed)
        
        # Use Nelder-Mead with adaptive parameters
        try:
            res = minimize(
                lambda x: objective(x, alpha=100.0),
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 1500,
                    'maxfev': 3000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            r = get_radius(res.x)
            if r > best_r:
                best_r = r
                best_x = res.x.copy()
        except Exception:
            pass

    # If we have a best candidate, do a final polish with higher sharpness
    if best_x is not None:
        try:
            res_polish = minimize(
                lambda x: objective(x, alpha=500.0),
                best_x,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'maxfev': 1000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            r_polish = get_radius(res_polish.x)
            if r_polish > best_r:
                best_x = res_polish.x.copy()
                best_r = r_polish
        except Exception:
            pass
            
    if best_x is None:
        # Fallback
        return random_init(0).reshape(N, 2)
        
    return best_x.reshape(N, 2)
# EVOLVE-BLOCK-END
