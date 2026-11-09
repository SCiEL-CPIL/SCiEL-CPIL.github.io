"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(demands, capacity):
    """
    Returns an n x n matrix of attractiveness for placing item j right after item i.
    
    Strategy: Complementarity graph with structural packing patterns.
    - Reward pairs that leave residual capacity as near-integer multiples of min item size
    - Combine with slack/tightness weighting and best-fit bin clustering
    - Pairs that don't fit get 1e-9
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    n = len(demands)
    demands = np.asarray(demands, dtype=np.float64)
    
    # Find minimum item size for structural pattern detection
    min_size = np.min(demands)
    
    # Best-Fit-Decreasing bin assignment for structural clustering
    order = np.argsort(-demands)
    
    bins = []  # list of lists of original indices
    bin_residuals = []  # remaining capacity per bin
    
    for idx in order:
        # Best-fit: find bin with smallest residual that still fits
        best_b = -1
        best_residual = float('inf')
        for b in range(len(bins)):
            if bin_residuals[b] >= demands[idx]:
                if bin_residuals[b] < best_residual:
                    best_residual = bin_residuals[b]
                    best_b = b
        if best_b != -1:
            bins[best_b].append(idx)
            bin_residuals[best_b] -= demands[idx]
        else:
            bins.append([idx])
            bin_residuals.append(capacity - demands[idx])
    
    # Build bin membership: bin_id for each item
    bin_of = np.full(n, -1, dtype=np.int64)
    for b, members in enumerate(bins):
        for idx in members:
            bin_of[idx] = b
    
    # Build heuristic matrix
    pair_sum = demands[:, None] + demands[None, :]
    fits = pair_sum <= capacity
    
    # Residual after placing pair
    residual = capacity - pair_sum
    
    # Structural pattern: reward residuals that are near-integer multiples of min_size
    # This encourages packing patterns where leftover space can accommodate small items
    if min_size > 0:
        # Distance to nearest integer multiple of min_size
        multiples = np.round(residual / min_size)
        nearest_mult = multiples * min_size
        struct_dist = np.abs(residual - nearest_mult)
        # Structural score: lower distance to structural grid -> higher score
        # Use exponential decay for smoothness
        struct_score = np.exp(-struct_dist / (0.5 * min_size))
    else:
        struct_score = np.ones((n, n))
    
    # Tightness score: prefer pairs that leave small residual (tighter pack)
    # Use inverse with offset to avoid division by zero
    tight_score = 1.0 / (1.0 + residual)
    
    # Same-bin bonus: items in the same BFD bin get a multiplier
    same_bin = (bin_of[:, None] == bin_of[None, :])
    same_bin_bonus = np.where(same_bin, 2.5, 1.0)
    
    # Combine scores: weighted combination of structural, tightness, and bin clustering
    # Weight structural pattern more heavily as it captures packing structure
    H = np.where(fits, (1.5 * struct_score + 1.0 * tight_score) * same_bin_bonus, 1e-9)
    
    # Ensure diagonal is not used (i != j), set to 1e-9
    np.fill_diagonal(H, 1e-9)
    
    # Ensure all values are positive and finite
    H = np.maximum(H, 1e-9)
    H = np.where(np.isfinite(H), H, 1e-9)
    
    return H
# EVOLVE-BLOCK-END
