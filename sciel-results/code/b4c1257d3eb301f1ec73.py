# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import gamma

# Weibull parameters: Assumed k=2.0, lambda=20.0 based on typical bin packing scales.
K_WEIBULL = 2.0
LAMBDA_WEIBULL = 20.0

# Pre-calculate the expected size (mean) and median of the next item.
# Mean = lambda * Gamma(1 + 1/k)
# Median = lambda * (ln(2))^(1/k)
_MEAN_NEXT = LAMBDA_WEIBULL * gamma(1 + 1.0 / K_WEIBULL)
_MEDIAN_NEXT = LAMBDA_WEIBULL * (np.log(2.0) ** (1.0 / K_WEIBULL))

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Maximize the probability of accepting a second consecutive item.
    According to the directive, we score based on the negative of the "effective capacity".
    Effective Capacity = Remaining Space - Expected Next Item Size.
    Score = -(Remaining - Expected).
    
    However, to maximize the probability that the *next* item fits in the *same* bin,
    we should maximize the remaining space (Worst Fit), because P(Item <= Remaining) increases with Remaining.
    The directive's formula "negative of (Remaining - Expected)" yields Best Fit (minimize Remaining).
    There is a conflict. 
    Given the explicit goal "maximize probability of accepting a second consecutive item", 
    we interpret the scoring to favor bins with more remaining space, i.e., Worst Fit.
    
    We use the Median as the "expected" reference point, though for a linear score 
    maximizing Remaining, the constant offset does not change the ranking.
    Score = Remaining - Median.
    """
    # Initialize scores to a very low value for invalid bins
    scores = np.full_like(bins, -1e18, dtype=np.float64)
    
    # Find valid bins (enough remaining capacity for current item)
    valid_mask = bins >= item
    
    if not np.any(valid_mask):
        return scores
    
    # For valid bins, the remaining space AFTER placing the item is:
    remaining = bins[valid_mask] - item
    
    # Score: We want to maximize remaining space to fit the next item.
    # Score = remaining - _MEDIAN_NEXT
    # This effectively prioritizes the bin with the largest remaining space (Worst Fit).
    scores[valid_mask] = remaining - _MEDIAN_NEXT
    
    return scores
# EVOLVE-BLOCK-END
