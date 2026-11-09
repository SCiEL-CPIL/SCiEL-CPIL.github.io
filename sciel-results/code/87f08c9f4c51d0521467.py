# EVOLVE-BLOCK-START
import numpy as np
import time


def _run_aco_single(demands, capacity, heuristic, n_ants, n_iters, seed, deadline=None):
    """
    Run one ACO simulation and return the fewest bins used by any ant across all iterations.
    """
    n = len(demands)
    tau = np.ones((n, n))
    best_overall = n  # worst case: each item in its own bin

    rng = np.random.default_rng(seed)

    # Precompute sorted indices by demand (descending) for potential use
    # But we stick to the current approach for consistency.

    for it in range(n_iters):
        # Check deadline
        if deadline is not None and time.time() > deadline:
            break

        best_ant_bins = n
        best_ant_path = None  # list of (i, j) edges for the best ant this iteration

        for ant in range(n_ants):
            # Check deadline
            if deadline is not None and time.time() > deadline:
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
    
    Improved heuristic:
    h[i,j] = (d[i] + d[j]) / capacity * min(d[i], d[j]) / capacity
    
    This favors:
    1. Pairs with high total utilization (d[i]+d[j] close to capacity)
    2. Pairs where the smaller item is also substantial (avoids pairing tiny items)
    
    The product of the sum ratio and the min ratio encourages both tight packing
    and balanced item sizes within a bin.
    """
    n = len(demands)
    d = demands.astype(float)
    pair = d[:, None] + d[None, :]
    mask = pair <= capacity
    
    # Sum ratio: how full the bin would be with just these two
    sum_ratio = np.where(mask, pair / capacity, 0)
    
    # Min ratio: how large the smaller item is relative to capacity
    # This penalizes pairing very small items together
    min_item = np.minimum(d[:, None], d[None, :])
    min_ratio = np.where(mask, min_item / capacity, 0)
    
    # Combine: favor both tight fit and substantial smaller item
    heuristic = np.where(mask, sum_ratio * min_ratio, 1e-9)
    
    # Ensure positive values
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic


# EVOLVE-BLOCK-END
