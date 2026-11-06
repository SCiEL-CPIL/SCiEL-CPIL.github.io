"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the bin with the highest score.
    
    Strategy: Next-Item Awareness heuristic combined with Best-Fit.
    We estimate the expected size of the next item and prefer bins that
    can accommodate both the current item and a typical next item, while
    still minimizing wasted space. This is a refinement of the best-fit
    approach to account for future items.
    """
    start_time = time.time()
    deadline = 150.0  # seconds
    
    # Ensure bins is a numpy array
    bins = np.asarray(bins, dtype=np.float64)
    
    # Filter bins that have enough remaining capacity
    mask = bins >= item
    if not np.any(mask):
        # No bin can hold the item - return all zeros (or negative infinity)
        # But the problem states "bins with enough remaining capacity"
        # If none exist, we should handle this gracefully.
        # According to the contract, the function returns scores for bins with enough capacity.
        # If no bins have enough capacity, the behavior is undefined but we should not crash.
        # Return zeros for all, which will likely lead to an error in the evaluator,
        # but the problem guarantees "each arriving item goes to the highest-scoring bin"
        # and implies there is always at least one bin with enough capacity.
        # To be safe, return a large negative number for bins that can't fit,
        # but the contract says "one score per bin with enough remaining capacity".
        # This suggests the input 'bins' already only contains bins with enough capacity?
        # No, the signature is priority(item, bins) where bins is the full array.
        # The docstring says "scores each bin with enough remaining capacity".
        # Let's re-read: "returns np.ndarray of one score per bin with enough remaining capacity".
        # This is ambiguous. Does it return an array of the same length, with -inf for invalid?
        # Or does it return a filtered array?
        # The example: return -(bins - item) returns an array of the same length.
        # So we should return an array of the same length.
        # Bins that can't hold the item should have a very low score (e.g., -np.inf).
        return np.where(mask, 0.0, -np.inf)
    
    # Calculate remaining capacity after placing the item
    remaining_after = bins - item
    
    # Estimate the expected size of the next item.
    # We can use the mean of the current items that fit in the bins,
    # or simply use the current item size as a baseline.
    # A more sophisticated approach: use the median of the remaining capacities
    # that are larger than the current item, as a proxy for typical next item size.
    # But to keep it simple and fast, let's use a weighted average of the current item
    # and the mean of the remaining capacities.
    
    # For now, let's try a simple heuristic:
    # Score = -|remaining_after - expected_next|
    # where expected_next is estimated from the distribution of bin remainders.
    
    # To estimate expected_next, we can use the mean of bins that are >= item
    # but not too much larger (to avoid outliers).
    # Or simply use the median of the remaining capacities.
    
    # Let's try: expected_next = median of remaining_after
    # But this might be circular.
    
    # Alternative: use the current item size as a proxy for the next item size,
    # and prefer bins that have enough space for the next item as well.
    # Score = -(remaining_after - item)  if remaining_after >= item else -(remaining_after)
    # This is like a "best-fit-decreasing" variant.
    
    # Let's try a smooth version:
    # Score = -max(0, remaining_after - item) - 0.5 * remaining_after
    # This penalizes bins that have too much extra space.
    
    # But to incorporate "next-item awareness", let's estimate the next item size
    # as the mean of the current item and the median of the remaining capacities.
    
    med_rem = np.median(remaining_after[mask])
    # Blend the current item size with the median remaining capacity
    expected_next = 0.5 * item + 0.5 * med_rem
    
    # Score: prefer bins where remaining_after is close to expected_next
    # Use a smooth distance metric
    distance = np.abs(remaining_after - expected_next)
    
    # But we also want to prefer bins with less remaining capacity (best-fit)
    # So combine both:
    score = -distance - 0.1 * remaining_after
    
    # Set invalid bins to -inf
    scores = np.where(mask, score, -np.inf)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline - 1.0:
        # Return best-fit as fallback
        return np.where(mask, -(bins - item), -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
