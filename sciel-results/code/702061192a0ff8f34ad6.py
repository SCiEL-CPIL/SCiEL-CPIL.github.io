"""Improved heuristic: tight-fit complementarity with global structure."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 320.0  # safety margin before 330s wall-clock kill


def heuristics(demands, capacity):
    """Item-pair attractiveness matrix for ant colony bin packing.

    Rewards pairs that fit (sum <= capacity) with high attractiveness that
    increases as the pair's combined size approaches capacity (tight fit =
    less wasted space). Adds a global structural component that favours
    pairing each item with its ideal complement in sorted order.
    """
    demands = np.asarray(demands, dtype=float)
    n = len(demands)

    if time.time() > _DEADLINE:
        # Fallback: simple but valid matrix
        pair = demands[:, None] + demands[None, :]
        return np.where(pair <= capacity, 1.0, 1e-9)

    # --- 1. Local tight-fit score ----------------------------------------
    # pair_sum[i,j] = demands[i] + demands[j]
    pair_sum = demands[:, None] + demands[None, :]
    fits = pair_sum <= capacity

    # Tightness: how close to capacity (0 = empty, 1 = exactly full)
    # Only meaningful where the pair fits.
    tightness = np.zeros((n, n), dtype=float)
    tightness[fits] = (pair_sum[fits] - 20.0) / (capacity - 20.0)
    # Clamp to [0, 1]
    tightness = np.clip(tightness, 0.0, 1.0)

    # Base attractiveness: strongly reward tight fits
    base = np.where(fits, tightness, 1e-9)

    # --- 2. Global structural complementarity ----------------------------
    # Sort items by size; for each item, its ideal partner is the largest
    # item that fits with it (two-pointer on sorted array).
    order = np.argsort(demands)  # ascending
    sorted_d = demands[order]

    # For each item i, find the ideal partner: largest j (in sorted order)
    # such that demands[i] + demands[j] <= capacity.
    # Use two-pointer: for smallest item, partner is largest that fits;
    # as item gets larger, partner index moves down.
    ideal_partner = np.full(n, -1, dtype=int)
    j_ptr = n - 1
    for i_idx in range(n):  # i_idx in sorted order (smallest to largest)
        while j_ptr > i_idx and sorted_d[i_idx] + sorted_d[j_ptr] > capacity:
            j_ptr -= 1
        if j_ptr >= i_idx:
            # The ideal partner in original indexing
            ideal_partner[order[i_idx]] = order[j_ptr]
        # Note: j_ptr is non-increasing as i_idx increases

    # Build a structural score: for each pair (i, j), measure how close j
    # is to i's ideal partner in sorted rank.
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(n)

    # Distance in sorted rank from ideal partner
    # For pair (i, j): if j == ideal_partner[i], distance = 0 (best)
    # Otherwise, distance = |rank[j] - rank[ideal_partner[i]]|
    ideal_rank = np.full(n, -1, dtype=int)
    for i in range(n):
        ip = ideal_partner[i]
        if ip >= 0:
            ideal_rank[i] = rank[ip]
        else:
            ideal_rank[i] = rank[i]  # fallback: self

    # For each i, compute rank distance to j for all j
    # rank_dist[i, j] = |rank[j] - ideal_rank[i]|
    rank_dist = np.abs(rank[None, :] - ideal_rank[:, None])

    # Structural bonus: Gaussian decay around ideal partner
    sigma = n / 4.0  # width of the structural preference
    structural = np.exp(-rank_dist ** 2 / (2.0 * sigma ** 2))

    # Only apply structural bonus where the pair actually fits
    structural = np.where(fits, structural, 0.0)

    # --- 3. Combine -------------------------------------------------------
    # Weighted combination: tight fit dominates, structure provides tie-breaking
    # and guides toward globally coherent packings.
    alpha = 0.7   # weight on tight-fit
    beta = 0.3    # weight on structural

    H = alpha * base + beta * structural

    # Ensure non-fitting pairs get the minimum value
    H = np.where(fits, H, 1e-9)

    # Ensure all values are positive and finite
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)

    # Normalise to [0, 1] range for numerical stability
    hmax = H.max()
    if hmax > 0 and np.isfinite(hmax):
        H = H / hmax
    H = np.where(H > 0, H, 1e-9)

    return H
# EVOLVE-BLOCK-END
