import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def heuristics(demands, capacity):
    """
    Heuristic matrix for ACO bin packing: attractiveness of placing item j after item i.
    
    Combines a refined product-to-waste-squared ratio with maximum weight matching boosts.
    The heuristic rewards:
    - Tight packing (low waste, squared penalty)
    - Large items (high product)
    - Balanced pairs (both items contribute significantly)
    - Globally optimal disjoint pairings (via matching boost)
    """
    start_time = time.time()

    d = np.asarray(demands, dtype=np.float64)
    n = len(d)

    if n == 0:
        return np.empty((0, 0))

    # Pairwise sums and products
    sums = d[:, None] + d[None, :]
    products = d[:, None] * d[None, :]

    # Feasibility mask
    fits = sums <= capacity

    # Waste = capacity - sum (non-negative where fits)
    waste = np.where(fits, capacity - sums, 1.0)

    # Smooth denominator: (waste + 1.0)^2
    denom = (waste + 1.0) ** 2

    # Balance factor: prefer pairs where both items are reasonably large
    min_items = np.minimum(d[:, None], d[None, :])
    safe_sums = np.where(sums > 0, sums, 1.0)
    balance = np.where(fits, min_items / safe_sums, 0.5)
    
    # Balance boost: stronger weight to prefer balanced pairs
    balance_boost = 1.0 + 1.0 * (balance - 0.5)

    # Tightness bonus: extra boost for very tight fits (waste <= 2)
    tightness_bonus = np.where(fits & (waste <= 2.0), 1.5, 1.0)

    # Base score: product * balance_boost * tightness_bonus / (waste + 1)^2
    base_scores = products * balance_boost * tightness_bonus / denom

    # Mask non-fitting pairs to 0 for matching
    base_scores_masked = np.where(fits, base_scores, 0.0)

    # Sanitize base scores
    if not np.all(np.isfinite(base_scores_masked)):
        base_scores_masked = np.where(np.isfinite(base_scores_masked), base_scores_masked, 0.0)

    # Find maximum weight matching of disjoint pairs
    # Use a time-efficient approach: only consider pairs where i < j
    # For n=120, this is 120x120 matrix, linear_sum_assignment is O(n^3) ~ 1.7M ops, fast enough
    
    matching_weights = np.zeros((n, n))
    # Vectorized fill for speed
    # We want matching_weights[i, j] = base_scores_masked[i, j] for all i != j where fits
    # Since base_scores_masked is symmetric (sums and products are symmetric), we can just copy
    matching_weights = base_scores_masked.copy()
    # Ensure diagonal is 0 (no self-pairs)
    np.fill_diagonal(matching_weights, 0.0)

    # Find optimal assignment (minimizes negative = maximizes positive)
    row_ind, col_ind = linear_sum_assignment(-matching_weights)

    # Build set of matched pairs
    matched_pairs = set()
    for r, c in zip(row_ind, col_ind):
        if r != c and matching_weights[r, c] > 0:
            matched_pairs.add((r, c))
            matched_pairs.add((c, r))

    # Construct final heuristic
    # Start with base scores
    H = base_scores.copy()

    # Boost matched pairs
    # Use a moderate boost factor to guide without overwhelming local heuristics
    # Inspiration 2 used 10x, Inspiration 1 used dynamic. Let's use a dynamic boost based on tightness
    # but capped to avoid instability.
    # Actually, let's stick to a strong fixed boost for simplicity and reliability, 
    # but slightly lower than 10x to allow some exploration, or keep 10x if it worked well.
    # The prompt says "strongest ideas". 10x worked for 1.053. 
    # Let's try a slightly adaptive boost: boost = 1 + 9 * (normalized_tightness)
    # But simpler is often better. Let's use 8.0 as a compromise.
    
    # Normalize base scores for boost calculation
    max_base = base_scores_masked.max()
    if max_base > 0:
        # For each matched pair, boost by factor depending on how good the pair is relative to max
        # This reinforces the "best" pairs more than just "any" matched pair
        for (i, j) in matched_pairs:
            # w is the normalized weight
            w = base_scores_masked[i, j] / max_base
            # Boost from 1.0 (if w=0) to 10.0 (if w=1)
            boost_factor = 1.0 + 9.0 * w
            H[i, j] = base_scores[i, j] * boost_factor

    # Mask non-fitting pairs to 1e-9
    H = np.where(fits, H, 1e-9)

    # Sanitize: replace any non-finite values
    if not np.all(np.isfinite(H)):
        H = np.where(np.isfinite(H), H, 1e-9)

    # Ensure all values are positive
    H = np.maximum(H, 1e-9)

    # Shape check
    if H.shape != (n, n):
        raise ValueError("Wrong shape")

    # Time guard
    elapsed = time.time() - start_time
    if elapsed > 290:
        return H

    return H
