"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the maximum radius for given centers."""
    # Distance to boundaries
    dist_bound = np.min(np.minimum(
        np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5),
        np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5)
    ))
    
    # Pairwise distances
    n = len(centers)
    if n < 2:
        return dist_bound / 2.0
    
    # Compute all pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, dist_bound)


def _negative_radius(centers):
    """Negative of radius for minimization."""
    return -_compute_radius(centers)


def _soft_negative_radius(centers, temperature=0.01):
    """Soft version using log-sum-exp for smoothness."""
    # Distance to boundaries
    dist_bound = np.min(np.minimum(
        np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5),
        np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5)
    ))
    
    # Pairwise distances
    n = len(centers)
    if n < 2:
        return -dist_bound / 2.0
    
    # Compute all pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    # Use soft min via log-sum-exp (inverted)
    # soft_min(d) = -logsumexp(-d/temperature) * temperature
    # We want min of dists, so:
    min_dists_sq = -temperature * np.log(np.sum(np.exp(-dists_sq / temperature)))
    
    # Convert to distance
    min_dist = np.sqrt(max(min_dists_sq, 0))
    
    radius = min(min_dist / 2.0, dist_bound)
    return -radius


def _init_grid(n):
    """Initialize with a square grid."""
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n]


def _init_random(n, seed=42):
    """Initialize with random positions."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(n, 2))


def _init_hybrid(n, seed=42):
    """Hybrid initialization: start from grid, add small random perturbations."""
    grid = _init_grid(n)
    rng = np.random.RandomState(seed)
    perturb = rng.normal(0, 0.02, size=(n, 2))
    centers = grid + perturb
    # Clip to valid range
    centers = np.clip(centers, -0.49, 0.49)
    return centers


def _coordinate_descent(centers, max_iters=500, tol=1e-8):
    """Coordinate-wise local refinement."""
    n = len(centers)
    current_radius = _compute_radius(centers)
    
    for it in range(max_iters):
        improved = False
        for i in range(n):
            for dim in range(2):
                # Try small steps in each direction
                for delta in [0.001, 0.0005, 0.0001, -0.001, -0.0005, -0.0001]:
                    new_center = centers[i].copy()
                    new_center[dim] += delta
                    # Check bounds
                    if -0.5 <= new_center[0] <= 0.5 and -0.5 <= new_center[1] <= 0.5:
                        centers[i] = new_center
                        new_radius = _compute_radius(centers)
                        if new_radius > current_radius + tol:
                            current_radius = new_radius
                            improved = True
                        else:
                            centers[i] -= delta  # revert
                    # Restore original
                    centers[i] = centers[i] - delta if not (new_radius > current_radius + tol) else centers[i]
        
        if not improved:
            break
    
    return centers, current_radius


def _local_optimize(centers, method='BFGS', max_iters=1000):
    """Local optimization using scipy."""
    x0 = centers.flatten()
    
    def objective(x):
        c = x.reshape(-1, 2)
        return _negative_radius(c)
    
    result = minimize(objective, x0, method=method, 
                      options={'maxiter': max_iters, 'gtol': 1e-6})
    
    return result.x.reshape(-1, 2), _compute_radius(result.x.reshape(-1, 2))


def construct():
    """Construct centers of 50 equal circles in unit square."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_centers = None
    best_radius = 0.0
    
    # Try multiple initialization strategies
    inits = [
        ('grid', _init_grid(N)),
        ('random', _init_random(N, seed=42)),
        ('hybrid', _init_hybrid(N, seed=42)),
        ('random2', _init_random(N, seed=123)),
        ('hybrid2', _init_hybrid(N, seed=123)),
    ]
    
    for name, centers in inits:
        if time.time() > deadline:
            break
        
        # First do coordinate descent
        centers_cd, radius_cd = _coordinate_descent(centers, max_iters=200)
        
        if radius_cd > best_radius:
            best_radius = radius_cd
            best_centers = centers_cd.copy()
        
        # Then do local optimization if time permits
        if time.time() < deadline - 20.0:
            centers_opt, radius_opt = _local_optimize(centers_cd, method='BFGS', max_iters=500)
            
            if radius_opt > best_radius:
                best_radius = radius_opt
                best_centers = centers_opt.copy()
        
        # Final polish with tighter tolerance
        if time.time() < deadline - 5.0:
            centers_polish, radius_polish = _local_optimize(best_centers, method='L-BFGS-B', max_iters=1000)
            
            if radius_polish > best_radius:
                best_radius = radius_polish
                best_centers = centers_polish.copy()
    
    # Ensure valid output
    if best_centers is None:
        best_centers = _init_grid(N)
    
    # Clip to ensure validity
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers
# EVOLVE-BLOCK-END
