# EVOLVE-BLOCK-START
import numpy as np
import time


def _run_aco_single(demands, capacity, heuristic, n_ants, n_iters, seed, time_limit=280.0):
    """
    Run one ACO simulation and return the fewest bins used by any ant across all iterations.
    This function is kept for compatibility with the seed program structure, 
    but the primary interface is heuristics().
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
        # Adaptive: reduce iterations if running slow
        elapsed_so_far = time.time() - start_time
        avg_time_per_iter = elapsed_so_far / (it + 1) if it > 0 else 0
        remaining_time = time_limit - elapsed_so_far
        if avg_time_per_iter > 0 and remaining_time / avg_time_per_iter < 1:
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
            if len(available) == 0:
                break
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
                    if len(available) == 0:
                        break
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
                    # Use softmax-like normalization for stability
                    max_p = np.max(probs)
                    if max_p > 0:
                        probs = probs / max_p
                    probs = probs / probs.sum()
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
            # Also reinforce diagonal slightly to encourage starting new bins wisely
            tau[np.diag_indices(n)] += gain * 0.1

    return best_overall


def heuristics(demands, capacity):
    """
    Compute the heuristic matrix.
    
    Strategy:
    1. Compute FFD bin assignment to get a baseline grouping.
    2. For each pair (i, j), if they fit (d_i + d_j <= C), calculate a base attractiveness
       based on how tightly they fit (smaller slack is better).
    3. If i and j are in the same FFD bin, boost the attractiveness by a factor.
    4. This creates a heuristic that favors tight fits, with a preference for fits
       that align with the FFD structure.
    """
    n = len(demands)
    
    # Sort items by size descending for FFD
    sorted_indices = np.argsort(-demands)
    sorted_demands = demands[sorted_indices]
    
    # Perform FFD assignment
    bins = []  # list of (bin_sum, [item_indices_in_original_order])
    item_to_bin = np.zeros(n, dtype=int)
    
    for idx in sorted_indices:
        placed = False
        for b_idx, (b_sum, b_items) in enumerate(bins):
            if b_sum + demands[idx] <= capacity:
                b_sum += demands[idx]
                b_items.append(idx)
                bins[b_idx] = (b_sum, b_items)
                item_to_bin[idx] = b_idx
                placed = True
                break
        if not placed:
            bins.append((demands[idx], [idx]))
            item_to_bin[idx] = len(bins) - 1
    
    # Build heuristic matrix
    # Base fit: exp(-slack / sigma)
    # sigma controls the sensitivity to slack.
    sigma = 15.0
    ffd_bonus = 2.5
    
    pair_sum = demands[:, None] + demands[None, :]
    fits = pair_sum <= capacity
    
    slack = capacity - pair_sum
    slack = np.where(fits, slack, capacity) # Set slack to capacity (large) if not fits
    
    base_attract = np.exp(-slack / sigma)
    
    # Apply FFD bonus
    same_bin = (item_to_bin[:, None] == item_to_bin[None, :])
    
    heuristic = base_attract.copy()
    heuristic[same_bin & fits] *= ffd_bonus
    
    # Ensure non-fitting pairs have very low value
    heuristic[~fits] = 1e-9
    
    # Ensure diagonal is positive (though it won't be used much)
    heuristic[np.diag_indices(n)] = 1e-9
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic


# EVOLVE-BLOCK-END
