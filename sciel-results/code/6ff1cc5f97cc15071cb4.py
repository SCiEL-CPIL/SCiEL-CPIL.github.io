import numpy as np
import time

# We assume capacity is 100 as per task description
CAPACITY = 100.0

# Deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 150.0
_SAFETY_MARGIN = 5.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Calculate remaining space
    remaining = bins - item
    
    # Initialize scores with -inf for invalid bins
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    # Find valid indices (bins with enough remaining capacity)
    valid_mask = remaining >= 0
    valid_idx = np.where(valid_mask)[0]
    
    if len(valid_idx) == 0:
        return scores

    # Structural change: Use a distribution-aware scoring that estimates
    # the future item distribution from the current state of all bins.
    # Key insight: The distribution of remaining spaces across ALL bins (not just valid)
    # gives information about the typical item sizes that have been placed.
    
    all_remaining = bins  # bins array contains remaining space
    valid_rems = remaining[valid_idx]
    
    # Estimate the "typical" item size from the distribution of occupied space
    # Occupied space = CAPACITY - remaining
    occupied = CAPACITY - all_remaining
    # Filter out empty bins (occupied near 0) to get a better estimate
    non_empty_occupied = occupied[occupied > 1.0]
    
    if len(non_empty_occupied) > 0:
        # Use the median occupied space as a proxy for typical item size
        # This is more robust than mean
        typical_item = np.median(non_empty_occupied)
    else:
        # Fallback to current item if no history
        typical_item = item
    
    # Score each valid bin by how well its new remainder aligns with the expected
    # distribution of future items.
    # 
    # New idea: Score based on the "fillability" of the new remainder.
    # A remainder r is "good" if it can be filled by a sequence of future items.
    # Approximate this by: how many of the observed item sizes could fit in r?
    # 
    # Even better: Use the empirical CDF of observed items to estimate
    # P(item <= r) for each possible remainder.
    #
    # For each valid bin i with new remainder r_i:
    #   score_i = P(future_item <= r_i) weighted by how "tight" the fit is
    #
    # Simpler structural approach: Use the variance of remainders as a signal.
    # If remainders are tightly clustered, we're in a stable regime.
    # If scattered, we should aim to reduce variance.
    
    # New structural approach: Score based on the "contribution to filling"
    # Place the item in the bin that minimizes the expected number of additional
    # items needed to fill that bin, assuming items follow the observed distribution.
    
    # For simplicity and speed, use a rank-based approach:
    # Score each bin by the rank of its new remainder among all current remainders.
    # Prefer bins where the new remainder is "typical" (median rank).
    
    all_rems_sorted = np.sort(all_remaining)
    
    # For each valid bin, compute where its new remainder would fall in the distribution
    # of all current remainders
    new_rems = valid_rems
    
    # Use np.searchsorted to find the rank of each new remainder
    ranks = np.searchsorted(all_rems_sorted, new_rems, side='left')
    max_rank = len(all_rems_sorted)
    
    # Normalize ranks to [0, 1]
    norm_ranks = ranks / max(max_rank, 1)
    
    # Score: prefer remainders that are close to the median (0.5)
    # This avoids both very small (dead) and very large (risky) gaps
    deviation_from_median = np.abs(norm_ranks - 0.5)
    
    # Additionally, penalize very small remainders (dead space)
    small_penalty = np.where(new_rems < typical_item * 0.3, 
                             (typical_item * 0.3 - new_rems) / typical_item, 
                             0)
    
    # Combine: score = -deviation - small_penalty
    valid_scores = -deviation_from_median - small_penalty
    
    scores[valid_idx] = valid_scores
    
    return scores
