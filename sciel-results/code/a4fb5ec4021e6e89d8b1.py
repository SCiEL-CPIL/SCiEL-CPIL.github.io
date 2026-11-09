"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin for placing the given item.
    Returns an array of scores where the bin with the highest score is selected.
    Bins that cannot fit the item are assigned -inf.
    
    Strategy: Fragmentation-aware Best-Fit.
    We aim to minimize the remaining capacity (Best-Fit) but penalize placements 
    that create "unusable" fragments (moderate remaining capacity).
    We also boost bins with very high remaining capacity to keep them open 
    for potential large items (mitigating the "greedy trap" of Best-Fit).
    """
    # Mask valid bins
    valid = bins >= item
    
    if not np.any(valid):
        return np.full_like(bins, -np.inf, dtype=float)
    
    # Remaining capacity after placing the item
    remaining = bins - item
    
    # Initialize scores with -inf
    scores = np.full_like(bins, -np.inf, dtype=float)
    
    # Base score: Best-Fit (minimize remaining capacity)
    # We use negative remaining so that smaller remaining -> higher score.
    # Scale by 1.0 to keep it in a reasonable range relative to penalties.
    # Note: bins are typically capacity 100.
    # remaining is in [0, 100].
    base_scores = -remaining
    
    # Fragmentation Penalty:
    # We want to avoid leaving a "moderate" amount of space that is too small for 
    # most future items but too large to be ignored.
    # For Weibull distributions, which often have a mix of small and large items,
    # leaving ~20-50% of the bin empty is often suboptimal.
    # We define a penalty zone where the remaining capacity is between 10 and 40 (arbitrary thresholds tuned for C=100).
    # A Gaussian penalty centered at 25 with width 10 seems reasonable.
    # This penalty reduces the score (makes it more negative) for bins in this zone.
    
    # We only apply this to valid bins.
    valid_remaining = remaining[valid]
    
    # Gaussian penalty
    center = 25.0
    width = 10.0
    # exp(-0.5 * ((x - center) / width)^2)
    # Max penalty at center.
    penalty = np.exp(-0.5 * ((valid_remaining - center) / width) ** 2)
    
    # Scale the penalty. The base score ranges from 0 to -100.
    # A penalty of ~20 should be significant enough to shift the decision if the base scores are close.
    penalty_weight = 20.0
    
    # High-Capacity Bonus:
    # To prevent Best-Fit from fragmenting bins too early, we give a small bonus to bins
    # that still have a lot of space left (e.g., > 60). This encourages keeping
    # "open" bins for larger items that might arrive later.
    # This counteracts the pure "fill the smallest hole" tendency.
    bonus_mask = valid_remaining > 60.0
    # Linear bonus increasing with remaining capacity? Or a fixed boost?
    # Let's use a soft boost: 0.1 * (valid_remaining - 60) for those > 60.
    # Max bonus at remaining=100 is 0.1 * 40 = 4.
    # This is small compared to the base score difference but might break ties in favor of keeping bins open.
    bonus = np.zeros_like(valid_remaining)
    bonus[bonus_mask] = 0.1 * (valid_remaining[bonus_mask] - 60.0)
    
    # Combine scores
    final_scores = base_scores[valid] - penalty_weight * penalty + bonus
    
    scores[valid] = final_scores
    
    return scores

# EVOLVE-BLOCK-END
