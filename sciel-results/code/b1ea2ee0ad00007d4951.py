"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score bins for online bin packing using Next-Item Awareness.
    
    Strategy: Estimate the expected size of the next arriving item from the Weibull
    distribution of item sizes, then score each valid bin based on how well its
    remaining space (after placing the current item) matches that predicted size.
    This reduces capacity waste by preferring bins where the leftover space is
    likely to be filled by the next item.
    
    We also incorporate a best-fit component to handle cases where no good match exists.
    """
    start_time = time.time()
    
    remaining = bins - item
    valid_mask = remaining >= 0
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(valid_mask):
        return scores
    
    valid_remaining = remaining[valid_mask]
    
    # Estimate expected next item size from Weibull distribution
    # Weibull with shape k=2, scale lambda=20 gives mean ~16.5 and typical items in [1, 50]
    # For capacity 100, this is a reasonable assumption for the test sequences
    # Expected value of Weibull(k, lambda) = lambda * Gamma(1 + 1/k)
    # For k=2, lambda=20: E[X] = 20 * Gamma(1.5) = 20 * 0.8862 = 17.72
    # But items are truncated to fit in capacity, so effective mean is slightly lower
    # Use a conservative estimate of ~15 for the expected next item size
    expected_next = 15.0
    
    # Score component 1: Next-item awareness
    # Reward bins whose remaining space is close to the expected next item size
    # Use a Gaussian-like penalty: high score when remaining ≈ expected_next
    # sigma controls how tightly we match; smaller sigma = stricter matching
    sigma = 8.0
    match_score = np.exp(-((valid_remaining - expected_next) ** 2) / (2 * sigma ** 2))
    
    # Score component 2: Best-fit (prefer smallest remaining space)
    # -log1p(remaining) gives higher scores for bins with less remaining space
    bestfit_score = -np.log1p(valid_remaining)
    
    # Combine: weight next-item awareness more heavily, but keep best-fit as tiebreaker
    # When remaining is very small (< 5), best-fit dominates (bin is nearly full)
    # When remaining is moderate, next-item awareness guides the choice
    blend_weight = 0.6  # weight for next-item awareness
    final_scores = blend_weight * match_score + (1 - blend_weight) * bestfit_score
    
    # Small bonus for bins that are nearly full to encourage closing them
    threshold = 5.0
    bonus = 0.05 * (valid_remaining < threshold).astype(float)
    final_scores += bonus
    
    scores[valid_mask] = final_scores
    
    # Safety check
    if time.time() - start_time > 1.0:
        pass
    
    return scores
# EVOLVE-BLOCK-END
