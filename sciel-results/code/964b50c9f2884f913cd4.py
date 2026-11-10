import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def heuristics(demands, capacity):
    """
    Heuristic matrix for ACO bin packing: attractiveness of placing item j after item i.
    
    Strategy:
    1. Compute a base tightness score for all feasible pairs using inverse waste.
    2. Identify globally optimal disjoint pairs via maximum weight matching.
    3. Boost these matched pairs significantly in the heuristic matrix to guide ACO.
    4. Apply a smooth gap penalty to further discourage wasteful pairings.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit

    d = np.asarray(demands, dtype=np.float64)
    n = len(d)

    if n == 0:
        return np.empty((0, 0))

    # Pairwise sums
    sums = d[:, None] + d[None, :]
    
    # Feasibility mask
    fits = sums <= capacity

    # Waste = capacity - sum (non-negative where fits)
    waste = np.where(fits, capacity - sums, 1.0)

    # Base score: 1 / (waste + 1)
    # This strongly rewards tight fits (waste=0 -> 1.0, waste=10 -> 0.09)
    base_scores = 1.0 / (waste + 1.0)
    
    # Mask non-fitting pairs
    base_scores = np.where(fits, base_scores, 0.0)

    # Sanitize base scores
    if not np.all(np.isfinite(base_scores)):
        base_scores = np.where(np.isfinite(base_scores), base_scores, 0.0)

    # Find maximum weight matching of disjoint pairs
    # We want to maximize the sum of weights, so we minimize negative weights.
    # Only consider feasible pairs. Note: linear_sum_assignment works on square matrices.
    # We create a weight matrix where only feasible pairs have positive weights.
    # To avoid matching i->i and double counting, we typically just let the algorithm handle it,
    # but we must ensure the weights for infeasible pairs are 0 (or very low) so they aren't picked.
    
    matching_weights = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if fits[i, j]:
                # Use base score as weight
                matching_weights[i, j] = base_scores[i, j]
                matching_weights[j, i] = base_scores[i, j]

    # Check time before expensive operation
    if time.time() > deadline:
        # Fallback to simple heuristic
        H = base_scores.copy()
        H = np.where(fits, H, 1e-9)
        H = np.maximum(H, 1e-9)
        return H

    # Find optimal assignment (minimizes negative = maximizes positive)
    row_ind, col_ind = linear_sum_assignment(-matching_weights)

    # Build set of matched pairs
    matched_pairs = set()
    for r, c in zip(row_ind, col_ind):
        if r != c and matching_weights[r, c] > 0:
            matched_pairs.add((r, c))
            matched_pairs.add((c, r))

    # Construct final heuristic: boost matched pairs significantly
    # Matched pairs get 10x boost to strongly guide ACO
    H = base_scores.copy()
    for (i, j) in matched_pairs:
        H[i, j] = base_scores[i, j] * 10.0

    # Apply a smooth gap penalty to further refine the scores
    # Penalty: multiply by (1 + waste / capacity)^(-gamma)
    # This adds a secondary gradient to prefer tighter fits even among non-matched items
    gamma = 2.0
    gap_penalty = (1.0 + waste / capacity) ** (-gamma)
    H = H * gap_penalty

    # Ensure non-fitting pairs are very low
    H = np.where(fits, H, 1e-9)

    # Sanitize: replace any non-finite values
    if not np.all(np.isfinite(H)):
        H = np.where(np.isfinite(H), H, 1e-9)

    # Ensure all values are positive (as per task requirements)
    H = np.maximum(H, 1e-9)

    # Shape check
    if H.shape != (n, n):
        raise ValueError("Wrong shape")

    # Time guard
    if time.time() > deadline:
        return H

    return H
