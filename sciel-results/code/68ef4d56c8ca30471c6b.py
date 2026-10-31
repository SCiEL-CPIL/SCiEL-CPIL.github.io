# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import gamma

# Weibull parameters for the generated sequences (shape k, scale lambda)
# Using k=2.0, lambda=20.0 as a reasonable default for bin packing with capacity 100
K_WEIBULL = 2.0
LAMBDA_WEIBULL = 20.0

def _weibull_cdf(x, k=K_WEIBULL, lam=LAMBDA_WEIBULL):
    """
    Compute the CDF of a Weibull distribution: P(X <= x) = 1 - exp(-(x/lam)^k)
    Returns 0 for x < 0, 1 for x >= 0 (since Weibull is non-negative).
    """
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x)
    mask = x >= 0
    result[mask] = 1.0 - np.exp(-np.power(x[mask] / lam, k))
    return result

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: For each valid bin, compute the probability that a future
    Weibull-distributed item will fit in the remaining space after placing
    the current item. Score = P(X_next <= remaining_space).
    This directly maximizes the probability of accepting a second consecutive item.
    """
    # Initialize scores to -infinity
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    # Find valid bins (enough remaining capacity for current item)
    valid_mask = bins >= item
    
    if not np.any(valid_mask):
        return scores
    
    # Calculate remaining space for valid bins after placing current item
    remaining = bins[valid_mask] - item
    
    # Score: probability that a future Weibull item fits in the remaining space
    # This is the CDF of the Weibull distribution evaluated at 'remaining'
    scores[valid_mask] = _weibull_cdf(remaining)
    
    return scores
# EVOLVE-BLOCK-END
