"""Online bin packing priority function."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Higher score is preferred (item goes to highest-scoring bin).
    
    Strategy: Best-Fit Decreasing variant.
    For online bin packing, Best-Fit (minimize remaining space after placement)
    is a strong baseline. We enhance it by considering the "tightness" of the fit.
    
    The score is designed so that bins that can accommodate the item and leave
    the least remaining capacity get the highest score.
    
    We use a simple heuristic: score = -remaining_capacity_after_placement.
    This is equivalent to Best-Fit.
    
    To potentially improve over pure Best-Fit, we can add a small bonus for
    bins that are already partially filled (to encourage consolidation), but
    Best-Fit already does this implicitly.
    
    Let's stick with a robust Best-Fit implementation:
    - Filter bins with enough capacity (bins >= item).
    - For those bins, compute remaining = bins - item.
    - Score = -remaining (so lower remaining -> higher score).
    - Bins that can't fit get a very low score (or -inf, but we need to handle
      the case where no bin can fit? The contract says "enough remaining capacity",
      but the function must return a score for ALL bins. Bins that can't fit
      should get a score so they are never chosen. We use -inf or a very negative number.
    """
    # Time guard: ensure we finish quickly. This function is O(n) where n is num bins.
    # Should be very fast.
    
    # Compute remaining capacity if item is placed
    # If bins < item, remaining would be negative. We want to disfavor these.
    
    # Create a score array
    # For bins that can fit: score = -(bins - item) = item - bins
    # For bins that can't fit: score = very negative
    
    scores = np.full(len(bins), -1e18)  # Very low score for infeasible bins
    feasible = bins >= item
    if np.any(feasible):
        scores[feasible] = item - bins[feasible]
    
    return scores
# EVOLVE-BLOCK-END
