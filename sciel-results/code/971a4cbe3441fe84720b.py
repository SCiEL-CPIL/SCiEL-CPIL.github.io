# EVOLVE-BLOCK-START
import numpy as np
import time


def _run_aco_single(demands, capacity, heuristic, n_ants, n_iters, seed, time_limit=280.0):
    """
    Run one ACO simulation and return the fewest bins used by any ant across all iterations.
    """
    n = len(demands)
    tau = np.ones((n, n))
    best_overall = n  # worst case: each item in its own bin

    rng = np.random.default_rng(seed)
    start_time = time.time()

    for it in range(n_iters):
        # Time guard: stop if we're running out of time
        if time.time() - start_time > time_limit:
            break

        best_ant_bins = n
        best_ant_path = None  # list of (i, j) edges for the best ant this iteration

        for ant in range(n_ants):
            if time.time() - start_time > time_limit:
                break
            used = np.zeros(n, dtype=bool)
            bins = 0
            current_sum = 0.0
            current_item = -1  # no item placed yet in current bin
            path = []

            # Place first item randomly
            # Pick a random unused item
            available = np.where(~used)[0]
            first = available[rng.integers(len(available))]
            used[first] = True
            current_sum = demands[first]
            current_item = first
            bins = 1

            # Now place remaining items
            while True:
                if time.time() - start_time > time_limit:
                    break
                # Find all unused items that fit
                remaining = np.where(~used)[0]
                if len(remaining) == 0:
                    break

                # Filter to those that fit
                slack = capacity - current_sum
                fitting = remaining[demands[remaining] <= slack]

                if len(fitting) == 0:
                    # Open new bin
                    bins += 1
                    current_sum = 0.0
                    current_item = -1
                    # Pick a random fitting item for the new bin (all fit since bin is empty)
                    # Actually, any unused item fits in an empty bin
                    available = np.where(~used)[0]
                    first = available[rng.integers(len(available))]
                    used[first] = True
                    current_sum = demands[first]
                    current_item = first
                    continue

                # Compute probabilities: tau[current_item, j] * heuristic[current_item, j]
                probs = tau[current_item, fitting] * heuristic[current_item, fitting]
                # Ensure no zeros cause issues; if all are zero, pick uniformly
                total = probs.sum()
                if total <= 0 or not np.isfinite(total):
                    idx = rng.integers(len(fitting))
                else:
                    probs = probs / total
                    # Use cumulative for sampling
                    cum = np.cumsum(probs)
                    cum[-1] = 1.0  # avoid floating point issues
                    r = rng.random()
                    idx = np.searchsorted(cum, r)
                    idx = min(idx, len(fitting) - 1)

                chosen = fitting[idx]
                used[chosen] = True
                path.append((current_item, chosen))
                current_sum += demands[chosen]
                current_item = chosen

            if bins < best_ant_bins:
                best_ant_bins = bins
                best_ant_path = path

        if best_ant_bins < best_overall:
            best_overall = best_ant_bins

        # Pheromone update
        tau *= 0.9
        if best_ant_bins > 0:
            gain = 1.0 / best_ant_bins
            if best_ant_path is not None:
                for i, j in best_ant_path:
                    tau[i, j] += gain

    return best_overall


def heuristics(demands, capacity):
    """
    Compute the heuristic matrix.
    
    Refined best-fit heuristic:
    - h[i,j] favors tight packing (high utilization of bin)
    - Uses a smooth function that penalizes leaving large gaps
    - Combines ratio-based attraction with a gap penalty
    
    The heuristic is: h[i,j] = (pair_sum / capacity)^alpha * exp(-beta * slack)
    where slack = capacity - pair_sum, and we only consider pairs that fit.
    
    This gives strong preference to near-full bins while remaining smooth.
    """
    n = len(demands)
    pair = demands[:, None] + demands[None, :]
    mask = pair <= capacity
    
    # Base ratio: how full the bin would be
    ratio = np.where(mask, pair / capacity, 0.0)
    
    # Slack: remaining space (lower is better)
    slack = np.where(mask, capacity - pair, capacity)
    
    # Combine: strong preference for high ratio, exponential decay for slack
    # alpha controls how strongly we favor tight fits
    # beta controls how sharply we penalize leftover space
    alpha = 4.0
    beta = 0.02
    
    # Compute heuristic with smooth combination
    # Use log-sum-exp style for numerical stability
    base = ratio ** alpha
    gap_penalty = np.exp(-beta * slack)
    
    heuristic = np.where(mask, base * gap_penalty, 1e-9)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic


# EVOLVE-BLOCK-END
