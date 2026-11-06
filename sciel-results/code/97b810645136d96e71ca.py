"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for online bin packing with a "Next-Item Awareness" heuristic.
    
    The evaluator picks the bin with the highest score among bins with enough remaining capacity.
    We want to minimize the number of bins used (excess over L1 lower bound).
    
    Strategy: 
    - Primary: Best-fit (minimize remaining space after placement)
    - Secondary: "Next-Item Awareness" - estimate the expected size of the next item
      from the current bin occupancy distribution and score bins based on how well
      their remaining space matches this predicted next item size.
    
    The idea is that if we can predict what the next item might look like, we can
    choose a bin where the remaining space is likely to be useful for the next item,
    rather than creating a gap that will be too small or too large for future items.
    
    For Weibull-distributed items with capacity 100, we estimate the expected next item
    size from the current state of the bins.
    """
    deadline = time.time() + 1.0  # 1 second budget with safety margin
    
    remaining = bins - item
    
    # Filter: only consider bins with enough remaining capacity
    valid_mask = remaining >= 0
    
    # If no valid bins, return scores that will result in no selection
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(valid_mask):
        return scores
    
    valid_remaining = remaining[valid_mask]
    n_valid = np.sum(valid_mask)
    
    # Estimate the expected next item size from the current bin occupancy
    # The bins array gives us the current fill level of each bin
    # We can estimate the distribution of item sizes from the gaps between
    # current fill levels and the capacity
    
    # For Weibull-distributed items, the expected value depends on the shape parameter
    # A simple heuristic: use the median of the remaining spaces as a proxy
    # for the typical "useful" space, which suggests the scale of items
    
    # Better approach: estimate the mean item size from the bins that are partially filled
    # If a bin has remaining space R, it suggests that the last item placed was 
    # approximately (capacity - R) minus the sum of previous items, but we don't have that.
    
    # Simpler heuristic: The expected next item size is related to the typical
    # remaining space in bins. If most bins have ~50 remaining, items are likely
    # around 50. If most have ~10 remaining, items are likely small.
    
    # Use the median of valid remaining spaces as an estimate of the "scale" of items
    # This is a rough proxy for the expected next item size
    median_remaining = np.median(valid_remaining)
    
    # The expected next item size is approximately the median remaining space,
    # but we need to be careful: if remaining is very small, the next item is likely small
    # If remaining is large, the next item could be large
    
    # For Weibull distributions with shape k and scale lambda:
    # E[X] = lambda * Gamma(1 + 1/k)
    # We can estimate lambda from the observed data
    
    # A practical approach: assume the next item size is drawn from a distribution
    # similar to the current item sizes. We can estimate the mean item size from
    # the bins.
    
    # The total capacity used across all valid bins is:
    # sum(bins[valid_mask]) = sum(item + fill_before) for each bin
    # But we don't know fill_before directly.
    
    # Alternative: Use the current item as a reference point
    # The expected next item is likely to be of similar magnitude to the current item
    # but adjusted by the distribution of remaining spaces
    
    # Let's use a weighted combination:
    # 1. Best-fit score: -remaining (prefer tight fits)
    # 2. Next-item awareness: how well the remaining space matches the expected next item
    
    # Estimate expected next item size:
    # Use the median of the remaining spaces, clamped to a reasonable range
    # If median_remaining is very small, the next item is likely small
    # If median_remaining is large, the next item could be larger
    
    # A simple model: expected_next_item = 0.5 * median_remaining + 0.5 * item
    # This blends the current item size with the typical remaining space
    expected_next_item = 0.5 * median_remaining + 0.5 * item
    
    # Ensure expected_next_item is positive and reasonable
    expected_next_item = max(expected_next_item, 1.0)
    expected_next_item = min(expected_next_item, 100.0)
    
    # Next-item awareness score:
    # We want the remaining space to be close to the expected next item size
    # This way, the next item can likely fit, reducing wasted space
    # Score = -|remaining - expected_next_item| (negative absolute difference)
    # Or use a Gaussian kernel for smoothness
    next_item_score = -np.abs(valid_remaining - expected_next_item)
    
    # Normalize the scores to a comparable scale
    # Best-fit score range: -remaining, so from -100 to 0
    # Next-item score range: -|remaining - expected|, so from -100 to 0
    
    # Combine the two scores with weights
    # Weight 1 for best-fit, weight 0.3 for next-item awareness
    base_score = -valid_remaining
    combined_score = 1.0 * base_score + 0.3 * next_item_score
    
    # Add a small quadratic penalty on remaining space to further favor tight fits
    capacity_estimate = 100.0
    quad_penalty = 0.01 * (valid_remaining ** 2) / capacity_estimate
    
    final_scores = combined_score - quad_penalty
    
    scores[valid_mask] = final_scores
    
    # Check deadline
    if time.time() > deadline:
        # Return whatever we have
        pass
    
    return scores
# EVOLVE-BLOCK-END
