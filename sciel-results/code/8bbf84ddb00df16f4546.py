import numpy as np

# We assume capacity is 100 as per task description
CAPACITY = 100.0

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

    # Strategy: Score each candidate bin by the negative count of other bins whose remainder is smaller than the resulting gap.
    # This approximates maximizing the likelihood that the residual space can be filled by future items.
    # If we place item in bin i, the new remainder is r_i = remaining[i].
    # We want to minimize the number of bins with remainder < r_i (among all valid bins).
    # Why? If many bins have remainders smaller than r_i, then r_i is "large" relative to the distribution of available spaces.
    # Wait, the directive says: "score each candidate bin by the negative count of other bins whose remainder is smaller than the resulting gap".
    # So score_i = - (count of j such that remaining[j] < remaining[i])
    # Higher score means fewer bins have smaller remainders.
    # This favors bins where the resulting gap is small (since fewer things are smaller than a small number).
    # This is essentially a rank-based Best Fit.
    # Let's verify:
    # If remaining[i] is very small (e.g., 1.0), few bins have remainder < 1.0. Count is small. Score is high (less negative).
    # If remaining[i] is large (e.g., 90.0), many bins have remainder < 90.0. Count is large. Score is low (more negative).
    # So this favors small remainders (Best Fit).
    # Is this better than pure Best Fit?
    # Pure Best Fit picks the absolute minimum.
    # This rank-based method might be more robust to noise or specific distribution shapes.
    # It effectively treats the remainders as a distribution and picks the bin that leaves a gap in the lower tail of the distribution.
    
    # Optimization:
    # For each valid index i, we want count(remaining[j] < remaining[i]) for j in valid_idx.
    # This can be done efficiently using sorting or searchsorted.
    
    valid_rems = remaining[valid_idx]
    
    # Sort the valid remainders
    sorted_rems = np.sort(valid_rems)
    
    # For each valid remainder r, find how many sorted_rems are strictly less than r.
    # searchsorted with side='left' gives the index where r would be inserted to maintain order,
    # which is the count of elements strictly less than r.
    # Note: searchsorted works on the array. We want the count for each r in valid_rems.
    counts = np.searchsorted(sorted_rems, valid_rems, side='left')
    
    # The score is -counts.
    # We assign these scores back to the valid indices.
    scores[valid_idx] = -counts
    
    return scores
