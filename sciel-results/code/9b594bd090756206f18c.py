"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Next-Item Awareness heuristic:
    - Estimate expected next item size from current bin occupancy distribution.
    - Score each valid bin by how well its post-placement remainder matches
      the predicted next item size, preferring remainders that can be fully
      utilized by the next item.
    """
    remaining = bins - item
    valid = remaining >= 0
    
    # Estimate expected next item size from the distribution of current
    # bin occupancies. The idea: items that have been placed recently
    # (reflected in current bin states) give us a sample of item sizes.
    # We use the mean of (capacity - bins) for non-empty bins as a proxy
    # for typical item size, but also consider the item itself.
    
    capacity = 100.0
    # Get current occupancies (used space in each bin)
    occupancies = bins
    
    # Estimate expected next item size:
    # Use a weighted combination of the current item and the mean occupancy
    # of bins that have been used. This adapts to the item size distribution.
    # For Weibull-distributed items, the mean is a reasonable central tendency.
    mean_occupancy = np.mean(occupancies)
    # Blend: 70% weight on current item (most recent signal), 30% on mean occupancy
    expected_next = 0.7 * item + 0.3 * mean_occupancy
    
    # For each valid bin, compute how well the remaining space after placing
    # the current item matches the expected next item size.
    # We want remaining ≈ expected_next so the next item fits well.
    # Score: negative of |remaining - expected_next|, with a bonus for
    # remaining being slightly larger (to accommodate variability).
    
    # Primary score: closeness of remainder to expected next item
    diff = remaining - expected_next
    # Use a smooth absolute value (soft-L1) for differentiability
    # |x| ≈ sqrt(x^2 + eps^2)
    eps_smooth = 1.0
    soft_abs = np.sqrt(diff ** 2 + eps_smooth ** 2)
    
    # Secondary: prefer bins with moderate utilization (not too empty, not too full)
    # This helps with overall packing density
    utilization = occupancies / capacity
    # Penalize very low utilization (< 0.1) and very high (> 0.9)
    util_penalty = np.where(utilization < 0.1, (0.1 - utilization) * 0.5,
                            np.where(utilization > 0.9, (utilization - 0.9) * 0.5, 0.0))
    
    # Tertiary: small bonus for remaining being slightly above expected_next
    # (to handle item size variability)
    slack_bonus = np.where(diff > 0, -0.1 * diff, 0.0)
    
    scores = -soft_abs - util_penalty + slack_bonus
    
    # Set invalid bins to very low score so they're never chosen
    scores = np.where(valid, scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
