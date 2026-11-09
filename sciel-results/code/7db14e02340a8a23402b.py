"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import minimize, LinearConstraint


def _score_centres(centres):
    """Compute the packing radius for a set of centres."""
    n = len(centres)
    if n == 0:
        return 0.0
    # Wall distance
    wall_min = 0.5 - np.max(np.abs(centres), axis=1)
    r_wall = float(np.min(wall_min))
    # Pairwise distances
    # Use vectorized pairwise distances for n=50 (1225 pairs)
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    # Exclude self-distances
    np.fill_diagonal(dists, np.inf)
    r_pair = float(np.min(dists)) / 2.0
    return min(r_wall, r_pair)


def _neg_score(centres):
    return -_score_centres(centres)


def _random_start(n, seed=42):
    rng = np.random.RandomState(seed)
    # Start with a simple grid perturbation
    m = int(np.ceil(n ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    grid = grid[:n]
    # Add small random perturbation
    grid = grid + rng.uniform(-0.01, 0.01, size=grid.shape)
    # Clip to valid range
    grid = np.clip(grid, -0.49, 0.49)
    return grid


def _local_optimize(centres, max_iter=200, time_limit=30):
    """Use scipy minimize with L-BFGS-B to maximize the minimum distance."""
    start_time = time.time()
    
    def objective(x):
        if time.time() - start_time > time_limit:
            raise TimeoutError()
        c = x.reshape(-1, 3)
        # Check validity
        if np.any(np.abs(c) > 0.5 + 1e-10):
            return 1e10
        return _neg_score(c)
    
    def gradient(x):
        c = x.reshape(-1, 3)
        n = len(c)
        # Compute gradient numerically (cheap for n=50)
        eps = 1e-7
        grad = np.zeros_like(x)
        f0 = objective(x)
        for i in range(len(x)):
            x2 = x.copy()
            x2[i] += eps
            f1 = objective(x2)
            grad[i] = (f1 - f0) / eps
            x2[i] -= 2 * eps
            f2 = objective(x2)
            grad[i] = (f1 - f2) / (2 * eps)
        return grad
    
    bounds = [(-0.5, 0.5)] * (3 * N)
    
    try:
        result = minimize(objective, centres.flatten(), method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': max_iter, 'maxfun': 1000})
        if result.success or result.fun < 0:
            return result.x.reshape(-1, 3)
        return centres
    except (TimeoutError, Exception):
        return centres


def _simulated_annealing(centres, iters=5000, time_limit=60):
    """Simple simulated annealing to improve the packing."""
    start_time = time.time()
    current = centres.copy()
    current_score = _score_centres(current)
    best = current.copy()
    best_score = current_score
    T = 0.01
    T_min = 1e-6
    cool = 0.999
    
    rng = np.random.RandomState(123)
    
    for it in range(iters):
        if time.time() - start_time > time_limit:
            break
        if T < T_min:
            break
        
        # Random perturbation
        idx = rng.randint(N)
        delta = rng.normal(0, T, size=3)
        candidate = current.copy()
        candidate[idx] += delta
        candidate[idx] = np.clip(candidate[idx], -0.5, 0.5)
        
        cand_score = _score_centres(candidate)
        delta_score = cand_score - current_score
        
        if delta_score > 0 or rng.random() < np.exp(delta_score / T):
            current = candidate
            current_score = cand_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
        
        T *= cool
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Start with a perturbed grid
    centres = _random_start(N, seed=42)
    
    # Try multiple random starts
    best_centres = centres
    best_score = _score_centres(centres)
    
    seeds = [42, 123, 456, 789, 101, 202, 303, 404]
    for seed in seeds:
        if time.time() > deadline:
            break
        c = _random_start(N, seed=seed)
        s = _score_centres(c)
        if s > best_score:
            best_centres = c
            best_score = s
    
    # Local optimization from the best start
    if time.time() < deadline - 20:
        opt_centres = _local_optimize(best_centres, max_iter=300, time_limit=30)
        opt_score = _score_centres(opt_centres)
        if opt_score > best_score:
            best_centres = opt_centres
            best_score = opt_score
    
    # Simulated annealing refinement
    if time.time() < deadline - 10:
        sa_centres = _simulated_annealing(best_centres, iters=10000, time_limit=40)
        sa_score = _score_centres(sa_centres)
        if sa_score > best_score:
            best_centres = sa_centres
            best_score = sa_score
    
    # Final local optimization
    if time.time() < deadline - 5:
        final_centres = _local_optimize(best_centres, max_iter=200, time_limit=15)
        final_score = _score_centres(final_centres)
        if final_score > best_score:
            best_centres = final_centres
            best_score = final_score
    
    # Ensure validity
    best_centres = np.clip(best_centres, -0.5, 0.5)
    
    return best_centres
# EVOLVE-BLOCK-END
