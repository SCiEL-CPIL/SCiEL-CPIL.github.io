"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import minimize


def _score_centres(centres):
    """Compute the packing radius for a set of centres."""
    n = len(centres)
    if n == 0:
        return 0.0
    # Wall distance
    wall_min = 0.5 - np.max(np.abs(centres), axis=1)
    r_wall = float(np.min(wall_min))
    # Pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    r_pair = float(np.min(dists)) / 2.0
    return min(r_wall, r_pair)


def _soft_min_score(centres, beta=100.0):
    """
    Compute a smooth surrogate for the packing radius using soft-min.
    This enables gradient-based optimization.
    """
    n = len(centres)
    if n == 0:
        return 0.0
    
    # Wall distances
    wall_dists = 0.5 - np.max(np.abs(centres), axis=1)
    
    # Pairwise distances / 2
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1)) / 2.0
    np.fill_diagonal(dists, np.inf)
    
    # Collect all constraint values
    all_vals = np.concatenate([wall_dists, dists[np.triu_indices(n, k=1)]])
    
    # Soft-min via log-sum-exp (for negative values to approximate min)
    # min(x) = -max(-x), and max(-x) ≈ (1/beta) * log(sum(exp(-beta*x)))
    neg_vals = -all_vals
    max_neg = np.max(neg_vals)
    if max_neg > 1e10:
        return 0.0
    log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_vals - max_neg)))
    soft_min = -log_sum_exp / beta
    
    return float(soft_min)


def _random_start(n, seed=42):
    rng = np.random.RandomState(seed)
    m = int(np.ceil(n ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    grid = grid[:n]
    grid = grid + rng.uniform(-0.02, 0.02, size=grid.shape)
    grid = np.clip(grid, -0.48, 0.48)
    return grid


def _local_optimize_smooth(centres, beta=100.0, max_iter=500, time_limit=30):
    """
    Optimize using smooth soft-min surrogate with L-BFGS-B.
    """
    start_time = time.time()
    
    def objective(x):
        if time.time() - start_time > time_limit:
            raise TimeoutError()
        c = x.reshape(-1, 3)
        if np.any(np.abs(c) > 0.5 + 1e-10):
            return 1e10
        return -_soft_min_score(c, beta=beta)
    
    bounds = [(-0.5, 0.5)] * (3 * N)
    
    try:
        result = minimize(objective, centres.flatten(), method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': max_iter, 'maxfun': 2000,
                                                  'ftol': 1e-12, 'gtol': 1e-8})
        if result.success or result.fun < 0:
            return result.x.reshape(-1, 3)
        return centres
    except (TimeoutError, Exception):
        return centres


def _high_precision_polish(centres, time_limit=20):
    """
    High-precision polish using iterative gradient steps on the exact objective.
    Uses a local search that moves the most constrained spheres.
    """
    start_time = time.time()
    current = centres.copy()
    current_score = _score_centres(current)
    best = current.copy()
    best_score = current_score
    
    step_size = 0.001
    n_iter = 0
    max_iter = 10000
    
    while time.time() - start_time < time_limit and n_iter < max_iter:
        n_iter += 1
        c = current
        n = len(c)
        
        # Find the minimum constraint
        wall_dists = 0.5 - np.max(np.abs(c), axis=1)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1)) / 2.0
        np.fill_diagonal(dists, np.inf)
        
        min_val = min(np.min(wall_dists), np.min(dists))
        
        # Find which spheres are most constrained
        # Move spheres that are at the minimum
        constrained_pairs = np.argwhere(dists <= min_val + 1e-10)
        constrained_walls = np.where(wall_dists <= min_val + 1e-10)[0]
        
        if len(constrained_pairs) == 0 and len(constrained_walls) == 0:
            break
        
        # Try to move one of the constrained spheres
        moves_made = False
        
        if len(constrained_walls) > 0:
            idx = constrained_walls[0]
            # Move away from wall
            for axis in range(3):
                if c[idx, axis] > 0:
                    c[idx, axis] -= step_size
                else:
                    c[idx, axis] += step_size
            c[idx] = np.clip(c[idx], -0.5, 0.5)
            moves_made = True
        
        if len(constrained_pairs) > 0:
            i, j = constrained_pairs[0]
            # Move i and j apart
            direction = c[i] - c[j]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction /= norm
                c[i] += direction * step_size * 0.5
                c[j] -= direction * step_size * 0.5
                c[i] = np.clip(c[i], -0.5, 0.5)
                c[j] = np.clip(c[j], -0.5, 0.5)
            moves_made = True
        
        if not moves_made:
            break
        
        new_score = _score_centres(c)
        if new_score > current_score:
            current = c
            current_score = new_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
            step_size = min(step_size * 1.1, 0.01)
        else:
            step_size *= 0.9
            if step_size < 1e-7:
                break
    
    return best


def _simulated_annealing(centres, iters=20000, time_limit=60):
    """Improved simulated annealing with better move strategies."""
    start_time = time.time()
    current = centres.copy()
    current_score = _score_centres(current)
    best = current.copy()
    best_score = current_score
    T = 0.02
    T_min = 1e-7
    cool = 0.9995
    
    rng = np.random.RandomState(123)
    
    for it in range(iters):
        if time.time() - start_time > time_limit:
            break
        if T < T_min:
            break
        
        # Choose move type
        move_type = rng.random()
        candidate = current.copy()
        
        if move_type < 0.7:
            # Single sphere move
            idx = rng.randint(N)
            scale = T * 2
            delta = rng.normal(0, scale, size=3)
            candidate[idx] += delta
        elif move_type < 0.85:
            # Pair move (swap-like)
            i, j = rng.choice(N, size=2, replace=False)
            scale = T
            candidate[i] += rng.normal(0, scale, size=3)
            candidate[j] += rng.normal(0, scale, size=3)
        else:
            # Small global perturbation
            scale = T * 0.5
            candidate += rng.normal(0, scale, size=candidate.shape)
        
        candidate = np.clip(candidate, -0.5, 0.5)
        
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
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Generate diverse initial configurations
    best_centres = None
    best_score = -1.0
    
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808]
    for seed in seeds:
        if time.time() > deadline - 30:
            break
        c = _random_start(N, seed=seed)
        s = _score_centres(c)
        if s > best_score:
            best_centres = c
            best_score = s
    
    # Smooth optimization from best start
    if time.time() < deadline - 40:
        opt_centres = _local_optimize_smooth(best_centres, beta=100.0, max_iter=500, time_limit=35)
        opt_score = _score_centres(opt_centres)
        if opt_score > best_score:
            best_centres = opt_centres
            best_score = opt_score
    
    # Simulated annealing refinement
    if time.time() < deadline - 30:
        sa_centres = _simulated_annealing(best_centres, iters=30000, time_limit=50)
        sa_score = _score_centres(sa_centres)
        if sa_score > best_score:
            best_centres = sa_centres
            best_score = sa_score
    
    # High-precision polish
    if time.time() < deadline - 10:
        polished = _high_precision_polish(best_centres, time_limit=25)
        polished_score = _score_centres(polished)
        if polished_score > best_score:
            best_centres = polished
            best_score = polished_score
    
    # Final smooth optimization pass with higher beta
    if time.time() < deadline - 5:
        final_opt = _local_optimize_smooth(best_centres, beta=200.0, max_iter=300, time_limit=15)
        final_score = _score_centres(final_opt)
        if final_score > best_score:
            best_centres = final_opt
            best_score = final_score
    
    # Final polish
    if time.time() < deadline - 2:
        final_polish = _high_precision_polish(best_centres, time_limit=10)
        final_polish_score = _score_centres(final_polish)
        if final_polish_score > best_score:
            best_centres = final_polish
            best_score = final_polish_score
    
    # Ensure validity
    best_centres = np.clip(best_centres, -0.5, 0.5)
    
    return best_centres
# EVOLVE-BLOCK-END
