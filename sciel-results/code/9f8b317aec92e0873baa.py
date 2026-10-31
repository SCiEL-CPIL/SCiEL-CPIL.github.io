import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # bins is an array of remaining capacities
    # We need to return scores for bins where bins >= item
    # The item goes to the highest-scoring bin.
    
    # Filter bins that can hold the item
    # But the function signature implies we return a score for *each* bin?
    # "returns np.ndarray of one score per bin with enough remaining capacity"
    # Wait, the description says: "returns np.ndarray of one score per bin with enough remaining capacity"
    # But the interface says: "priority(item: float, bins: np.ndarray) -> np.ndarray of one score per bin"
    # Usually, in these FunSearch/EoH contracts, you return a score for ALL bins, and the caller picks the max among valid ones, or you return -inf for invalid ones.
    # The seed program `return -(bins - item)` returns a value for every bin.
    # If a bin doesn't have enough capacity, `bins - item` is negative, so `-(bins - item)` is positive.
    # This would incorrectly select an invalid bin if its score is higher than valid bins.
    # However, the seed program is the baseline (score 1.0). It must be working.
    # Let's re-read carefully: "scores each bin with enough remaining capacity".
    # This implies the caller might filter, or the scores for invalid bins should be very low.
    # If I return `-(bins - item)`, and `bins < item`, then `bins - item` is negative, so score is positive.
    # If a valid bin has `bins - item = 1`, score is -1.
    # An invalid bin with `bins - item = -10` has score 10.
    # The invalid bin would be chosen!
    # So the seed program `return -(bins - item)` is likely flawed for invalid bins UNLESS the caller filters.
    # BUT, the task says "The evaluator measures the raw value...".
    # If the seed is 1.0, it is a valid baseline.
    # Perhaps the `bins` array passed in ONLY contains bins with enough capacity?
    # "priority(item, bins) ... scores each bin with enough remaining capacity"
    # This phrasing suggests `bins` might already be filtered, OR I am expected to handle the filtering by returning -inf for invalid bins.
    # Let's look at the standard contract for these tasks. Usually, you return a score for every bin in the input array, and you must ensure invalid bins have very low scores (e.g., -np.inf).
    # The seed `return -(bins - item)` does NOT do this.
    # If the seed works, then either:
    # 1. The `bins` input is pre-filtered to only contain valid bins.
    # 2. The evaluator ignores invalid bins somehow.
    # 3. The seed is actually bad but it's the baseline.
    # Given "score 1.0000", it's the baseline.
    # If I assume `bins` contains ALL bins (valid and invalid), I MUST mask invalid bins.
    # If I assume `bins` contains ONLY valid bins, I don't need to mask.
    # The prompt says: "scores each bin with enough remaining capacity".
    # This is ambiguous. Does it mean "The function scores the bins that have enough capacity" (implying the input is filtered) or "The function should assign scores to bins, considering only those with enough capacity" (implying I should mask)?
    # Standard practice in these optimization tasks: Return -np.inf for infeasible bins.
    # Let's modify the seed to be safe: `scores = -(bins - item); scores[bins < item] = -np.inf`.
    # If the input is already filtered, this has no effect (or rather, `bins < item` is all False).
    # If the input is not filtered, this prevents crashes/invalid selections.
    
    # Strategy: Hybrid Best/Worst Fit
    # If item > 50, use Best Fit (minimize remaining space).
    # If item <= 50, use Worst Fit (maximize remaining space).
    
    remaining = bins - item
    valid = bins >= item
    
    # Best Fit score: -remaining (higher score for smaller remaining)
    # Worst Fit score: remaining (higher score for larger remaining)
    
    scores = np.zeros_like(bins, dtype=float)
    
    if item > 50:
        # Best Fit
        scores = -remaining
    else:
        # Worst Fit
        scores = remaining
        
    # Mask invalid bins
    scores[~valid] = -np.inf
    
    return scores
