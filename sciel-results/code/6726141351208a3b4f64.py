# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import gamma

# Weibull parameters for the generated sequences (shape k, scale lambda)
K_WEIBULL = 2.0
LAMBDA_WEIBULL = 20.0

def _expected_next_item_size() -> float:
    """
    Estimate the expected size of the next item.
    For a Weibull distribution, E[X] = lambda * Gamma(1 + 1/k).
    """
    return LAMBDA_WEIBULL * gamma(1 + 1.0 / K_WEIBULL)

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Minimize the "effective waste" by scoring bins based on how well
    the remaining space matches the expected size of the next item.
    """
    expected_next = _expected_next_item_size()
    
    # Initialize scores to -infinity
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    # Find valid bins
    valid_mask = bins >= item
    
    if not np.any(valid_mask):
        # No valid bins; return all -inf (or any values, but -inf is safe)
        return scores
    
    # Calculate remaining space for valid bins
    remaining = bins[valid_mask] - item
    
    # Score: negative of (remaining - expected_next_size)
    # This encourages bins where remaining is close to expected_next_size.
    # If remaining is much larger, score is low (negative).
    # If remaining is close to expected_next_size, score is high (near 0 or positive).
    # If remaining is smaller than expected_next_size, score is positive.
    scores[valid_mask] = -(remaining - expected_next)
    
    return scores
# EVOLVE-BLOCK-END
