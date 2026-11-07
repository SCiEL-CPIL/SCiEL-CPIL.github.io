"""Heuristic for ACO bin packing: adaptive selection within tightness+scarcity family."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(demands, capacity):
    demands = np.asarray(demands, dtype=float)
    n = len(demands)
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit

    # Precompute common quantities
    pair_sum = demands[:, None] + demands[None, :]
    fits = pair_sum <= capacity
    waste = capacity - pair_sum
    degree = fits.sum(axis=0).astype(float)
    degree_safe = np.maximum(degree, 1.0)
    scarcity = 1.0 / degree_safe  # shape (n,)

    eps = 1e-6

    # Candidate heuristics within the tightness+scarcity family
    candidates = []

    # 1. Base: tightness * scarcity (current program)
    tightness = np.where(fits, 1.0 / (waste + eps), 0.0)
    h1 = tightness * scarcity[None, :]
    candidates.append(h1)

    # 2. Sharper tightness: (1/waste)^2 * scarcity
    tightness2 = np.where(fits, 1.0 / (waste + eps)**2, 0.0)
    h2 = tightness2 * scarcity[None, :]
    candidates.append(h2)

    # 3. Tightness * scarcity with size-similarity bonus (from inspiration 1)
    d_i = demands[:, None]
    d_j = demands[None, :]
    denom = np.maximum(d_i * d_j, eps)
    sim = (d_i * d_j) / (denom + 1.0)  # in (0,1), ~1 when sizes similar
    h3 = tightness * scarcity[None, :] * (0.5 + 0.5 * sim)
    candidates.append(h3)

    # 4. Tightness * scarcity with fill-ratio bonus
    fill_ratio = pair_sum / capacity
    fill_bonus = 1.0 / (1.0 + np.abs(fill_ratio - 1.0) + eps)
    h4 = tightness * scarcity[None, :] * fill_bonus
    candidates.append(h4)

    # 5. Combined: tightness * scarcity * sim * fill_bonus
    h5 = tightness * scarcity[None, :] * (0.5 + 0.5 * sim) * fill_bonus
    candidates.append(h5)

    # 6. Exponential tightness: exp(-waste/scale) * scarcity
    for scale in [10.0, 20.0]:
        h_exp = np.where(fits, np.exp(-waste / scale), 0.0)
        h = h_exp * scarcity[None, :]
        candidates.append(h)

    # Clean all candidates
    clean_candidates = []
    for h in candidates:
        h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
        clean_candidates.append(h)

    # If time is tight, just return the base heuristic
    if time.time() > deadline - 30:
        return clean_candidates[0]

    # Fast ACO simulation to evaluate candidates
    eval_seeds = [42, 123, 999]
    best_h = clean_candidates[0]
    best_score = float('inf')

    for idx, H in enumerate(clean_candidates):
        if time.time() > deadline - 20:
            break

        scores = []
        for seed in eval_seeds:
            if time.time() > deadline - 15:
                break
            bins_used = _simulate_aco_fast(demands, capacity, H, seed=seed, ants=10, iters=10)
            scores.append(bins_used)

        if len(scores) == len(eval_seeds):
            avg_score = np.mean(scores)
            if avg_score < best_score:
                best_score = avg_score
                best_h = H

    # Final sanity check
    best_h = np.where(np.isfinite(best_h) & (best_h > 0), best_h, 1e-9)
    return best_h


def _simulate_aco_fast(demands, capacity, H, seed=42, ants=10, iters=10):
    """Fast ACO simulation for heuristic evaluation."""
    np.random.seed(seed)
    n = len(demands)
    W = demands

    tau = np.ones((n, n))
    min_bins_global = float('inf')

    for it in range(iters):
        min_bins_this_iter = float('inf')

        for ant in range(ants):
            used = np.zeros(n, dtype=bool)
            bins_used = 0
            items_left = n

            while items_left > 0:
                bins_used += 1
                unused_indices = np.where(~used)[0]
                if len(unused_indices) == 0:
                    break

                # Random fitting item to start
                start_idx = unused_indices[np.random.randint(len(unused_indices))]
                used[start_idx] = True
                items_left -= 1
                current = start_idx
                current_load = W[current]

                while True:
                    remaining = capacity - current_load
                    fits = (~used) & (W <= remaining)

                    if not np.any(fits):
                        break

                    fit_indices = np.where(fits)[0]
                    probs = tau[current, fit_indices] * H[current, fit_indices]
                    probs = np.maximum(probs, 1e-12)
                    probs /= probs.sum()

                    next_idx = fit_indices[np.random.choice(len(fit_indices), p=probs)]
                    used[next_idx] = True
                    items_left -= 1
                    current_load += W[next_idx]
                    current = next_idx

            min_bins_this_iter = min(min_bins_this_iter, bins_used)

        # Pheromone update
        tau *= 0.9
        tau += 1.0 / min_bins_this_iter

        min_bins_global = min(min_bins_global, min_bins_this_iter)

    return min_bins_global
# EVOLVE-BLOCK-END
