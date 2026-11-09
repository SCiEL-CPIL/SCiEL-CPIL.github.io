"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Higher score = better choice.
    
    Strategy: Combine multiple heuristics for online bin packing.
    - Prefer bins where the item fits with minimal waste (best-fit-like)
    - But also consider global distribution to avoid fragmenting bins
    - Use a weighted combination that balances local fit quality with
      maintaining some bins with large contiguous free space
    """
    # Track time to ensure we don't exceed budget
    t0 = time.time()
    
    # Get remaining capacity in each bin
    remaining = bins - item
    
    # Only consider bins where the item fits
    valid_mask = remaining >= 0
    
    # If no valid bins, return zeros (shouldn't happen in normal operation)
    if not np.any(valid_mask):
        return np.zeros_like(bins, dtype=float)
    
    # For valid bins, compute multiple scoring components
    
    # 1. Best-fit score: prefer bins with least remaining space after placing item
    #    (negative remaining, so higher = less remaining = better fit)
    best_fit = -remaining[valid_mask]
    
    # 2. Normalize best_fit to [0, 1] range for weighting
    if len(best_fit) > 0:
        bf_min = np.min(best_fit)
        bf_max = np.max(best_fit)
        if bf_max > bf_min:
            best_fit_norm = (best_fit - bf_min) / (bf_max - bf_min)
        else:
            best_fit_norm = np.ones_like(best_fit)
    else:
        best_fit_norm = np.array([])
    
    # 3. Consider the current bin utilization (fullness)
    #    Bins that are more full have less "wasted" space
    #    But we want to avoid creating many partially-filled bins
    utilization = 1.0 - remaining[valid_mask] / 100.0  # assume capacity 100
    
    # 4. Combine: primarily best-fit, with a small penalty for very full bins
    #    to maintain some bins with large free space for future large items
    #    This is a compromise between best-fit and first-fit-decreasing-like behavior
    
    # Weighted score: 80% best-fit, 20% penalty for being too full
    # The penalty encourages keeping some bins less full
    fullness_penalty = utilization ** 3  # cubic to only penalize very full bins heavily
    
    score = 0.8 * best_fit_norm - 0.2 * fullness_penalty
    
    # Build the full score array (invalid bins get very negative score)
    scores = np.full_like(bins, -1e9, dtype=float)
    scores[valid_mask] = score
    
    # Safety check: if we're taking too long, return simple best-fit
    if time.time() - t0 > 0.01:  # 10ms per call is already generous
        scores = np.full_like(bins, -1e9, dtype=float)
        scores[valid_mask] = -remaining[valid_mask]
    
    return scores
# EVOLVE-BLOCK-END
