"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _score_centres(centres):
    """Compute the packing radius for a set of centres."""
    n = len(centres)
    if n == 0:
        return 0.0
    wall_min = 0.5 - np.max(np.abs(centres), axis=1)
    r_wall = float(np.min(wall_min))
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    r_pair = float(np.min(dists)) / 2.0
    return min(r_wall, r_pair)


def _soft_min_score(centres, beta=100.0):
    """
    Smooth surrogate for the packing radius using soft-min (log-sum-exp).
    Returns a differentiable approximation to -min(distances/2, wall_distances).
    """
    n = len(centres)
    if n == 0:
        return 0.0

    # Wall distances
    wall_min = 0.5 - np.max(np.abs(centres), axis=1)

    # Pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    # Only upper triangle to avoid double counting
    iu = np.triu_indices(n, k=1)
    pair_dists_sq = dists_sq[iu]

    # Soft-min over all constraints: pairwise distances and wall distances
    # We want to maximize the minimum, so minimize the negative
    # Use log-sum-exp for soft-min: soft_min(x) = -log(sum(exp(-x/beta))) * beta
    # For pairwise: values are dists/2, for walls: wall_min
    all_vals = np.concatenate([pair_dists_sq * 0.5, wall_min])  # squared for now
    # Actually let's work with distances directly for clarity
    # Recompute pairwise distances (not squared)
    pair_dists = np.sqrt(pair_dists_sq)

    all_vals = np.concatenate([pair_dists * 0.5, wall_min])

    # Soft min: -beta * log(mean(exp(-x/beta)))
    # For numerical stability
    min_val = np.min(all_vals)
    exp_terms = np.exp(-(all_vals - min_val) / beta)
    soft_min_val = min_val - beta * np.log(np.mean(exp_terms))

    return -soft_min_val  # negative for minimization


def _halton_init(n, seed=42):
    """Generate n points in [0,1]^3 using Halton low-discrepancy sequence."""
    rng = np.random.RandomState(seed)

    def halton(index, base):
        """Generate the index-th Halton sequence value in [0,1) for given base."""
        f = 1.0
        result = 0.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result

    points = np.zeros((n, 3))
    for i in range(n):
        for dim, base in enumerate([2, 3, 5]):
            points[i, dim] = halton(i + 1, base)

    # Map from [0,1]^3 to [-0.5, 0.5]^3
    points = points - 0.5

    # Add small random perturbation to break any residual symmetry
    points += rng.uniform(-0.005, 0.005, size=points.shape)

    # Clip to valid range
    points = np.clip(points, -0.49, 0.49)
    return points


def _local_optimize_soft(centres, start_time, deadline, beta_start=50.0, beta_end=2000.0,
                          max_iter=500):
    """
    Use scipy L-BFGS-B with a soft-min surrogate and annealing on beta.
    """
    current = centres.copy()
    n_params = 3 * N
    bounds = [(-0.5, 0.5)] * n_params

    # Anneal beta from start to end over the optimization
    n_stages = 5
    betas = np.geomspace(beta_start, beta_end, n_stages)

    for beta in betas:
        if time.time() > deadline - 5:
            break

        def objective(x):
            c = x.reshape(-1, 3)
            if np.any(np.abs(c) > 0.5 + 1e-10):
                return 1e10
            return _soft_min_score(c, beta=beta)

        try:
            result = minimize(objective, current.flatten(), method='L-BFGS-B',
                              bounds=bounds, options={'maxiter': max_iter, 'maxfun': 5000})
            if result.fun < 1e9:
                current = result.x.reshape(-1, 3)
        except Exception:
            pass

    return current


def _local_optimize_exact(centres, start_time, deadline, max_iter=200):
    """
    Final polish using exact (non-smooth) score with coordinate descent.
    """
    current = centres.copy()
    n = N

    def score(c):
        return _score_centres(c)

    current_score = score(current)

    # Coordinate descent with adaptive step
    step = 0.005
    for iteration in range(max_iter):
        if time.time() > deadline - 3:
            break

        improved = False
        for i in range(n):
            for dim in range(3):
                # Try moving in both directions
                for direction in [1, -1]:
                    trial = current.copy()
                    trial[i, dim] += direction * step
                    trial[i, dim] = np.clip(trial[i, dim], -0.5, 0.5)
                    trial_score = score(trial)
                    if trial_score > current_score + 1e-12:
                        current = trial
                        current_score = trial_score
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

        if not improved:
            # Reduce step size
            step *= 0.8
            if step < 1e-6:
                break
        else:
            # Keep step or slightly increase
            step = min(step * 1.1, 0.01)

    return current


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    # Try multiple Halton-based initializations with different seeds
    best_centres = None
    best_score = -1.0

    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
    for seed in seeds:
        if time.time() > deadline - 30:
            break
        c = _halton_init(N, seed=seed)
        s = _score_centres(c)
        if s > best_score:
            best_centres = c
            best_score = s

    # Soft-min optimization with beta annealing
    if time.time() < deadline - 20:
        opt_centres = _local_optimize_soft(best_centres, start_time, deadline - 20,
                                            beta_start=50.0, beta_end=2000.0, max_iter=300)
        opt_score = _score_centres(opt_centres)
        if opt_score > best_score:
            best_centres = opt_centres
            best_score = opt_score

    # Exact coordinate descent polish
    if time.time() < deadline - 10:
        final_centres = _local_optimize_exact(best_centres, start_time, deadline - 10, max_iter=500)
        final_score = _score_centres(final_centres)
        if final_score > best_score:
            best_centres = final_centres
            best_score = final_score

    # Ensure validity
    best_centres = np.clip(best_centres, -0.5, 0.5)

    return best_centres
# EVOLVE-BLOCK-END
