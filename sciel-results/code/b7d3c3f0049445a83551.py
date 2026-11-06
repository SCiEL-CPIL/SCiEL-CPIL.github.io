import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Identify feasible bins (enough remaining capacity)
    feasible = bins >= item
    
    n = len(bins)
    result = np.full(n, -np.inf)
    
    if not np.any(feasible):
        return result
    
    feasible_indices = np.where(feasible)[0]
    new_rem = bins[feasible] - item
    
    # Rank-based scoring: negative count of bins with strictly smaller remainder
    # This favors bins that, after placing the item, have a remainder that is
    # relatively large compared to other feasible bins' remainders (Worst-Fit-like)
    # but in a rank/distribution-aware way.
    new_rem_sorted = np.sort(new_rem)
    counts = np.searchsorted(new_rem_sorted, new_rem, side='left')
    
    # We want to maximize the score (highest scoring bin is chosen).
    # Using positive counts means bins with larger remainders (relative to others) get higher scores.
    # Add a tiny negative index penalty to break ties deterministically (prefer lower index).
    idx = feasible_indices.astype(np.float64)
    scores = counts.astype(np.float64) - 1e-9 * idx
    
    result[feasible_indices] = scores
    
    return result
