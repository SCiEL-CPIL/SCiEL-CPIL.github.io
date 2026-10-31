"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy import stats

# Weibull parameters calibrated for Weibull-distributed items with capacity 100
# Using shape=1.5 (slightly right-skewed) and scale=40.0 (typical item size)
WEIBULL_SHAPE = 1.5
WEIBULL_SCALE = 40.0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin with enough remaining capacity using a Weibull survival-based
    auction strategy. Each bin's remaining capacity is a "bid" for the item.
    Score = -S(remainder), where S is the Weibull survival function.
    
    This directly minimizes the probability that the leftover space becomes
    permanently unusable waste, prioritizing remainders that fall within the
    high-density region of the Weibull distribution (where future items are
    most likely to fit).
    """
    capacity = 100.0
    
    # Calculate remaining capacity if item is placed
    remainders = bins - item
    
    # Mask for valid bins
    valid_mask = remainders >= 0
    
    # Initialize scores to a very low value for invalid bins
    scores = np.full_like(bins, -1e9, dtype=float)
    
    if not np.any(valid_mask):
        return scores
    
    valid_remainders = remainders[valid_mask]
    
    # Weibull survival function: S(r) = exp(-(r/scale)^shape)
    # For small remainders near 0, S(r) is high (likely to be wasted)
    # For remainders in the high-density region, S(r) is moderate
    # For very large remainders, S(r) approaches 0 (safe, but potentially wasteful of capacity)
    
    # Compute -S(r) to maximize: we want the highest score (closest to 0 from negative side)
    # This means we prefer remainders where S(r) is smallest, i.e., large remainders
    # But we also want to avoid creating huge gaps that will never be filled
    
    # The Weibull survival function naturally penalizes small remainders heavily
    # and rewards remainders that are large enough that future items are unlikely to fit
    # (since if future items are unlikely to fit, the space isn't wasted)
    
    # However, we also want to consolidate small items into full bins
    # So we add a small bonus for remainders near 0 (full bins)
    
    # Compute the Weibull survival
    r_scaled = valid_remainders / WEIBULL_SCALE
    survival = np.exp(-np.power(r_scaled, WEIBULL_SHAPE))
    
    # Base score: negative survival (higher is better)
    base_scores = -survival
    
    # Add a small consolidation bonus for near-full bins
    # This helps pack small items together without overwhelming the survival-based logic
    consolidation_bonus = 0.1 * np.exp(-valid_remainders / 5.0)
    
    valid_scores = base_scores + consolidation_bonus
    
    scores[valid_mask] = valid_scores
    
    return scores
# EVOLVE-BLOCK-END
