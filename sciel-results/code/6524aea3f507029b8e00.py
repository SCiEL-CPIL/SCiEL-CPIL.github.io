# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import gamma
from scipy.stats import weibull_min

# Weibull parameters for the generated sequences (shape k, scale lambda)
# Using k=2.0, lambda=20.0 as a reasonable default for Weibull sequences
# with capacity 100, giving mean item size ~17.7

K_WEIBULL = 2.0
LAMBDA_WEIBULL = 20.0
CAPACITY = 100.0

def _weibull_cdf(x: np.ndarray) -> np.ndarray:
    """
    Compute the CDF of the Weibull distribution for given values x.
    P(X <= x) = 1 - exp(-(x/lambda)^k)
    """
    x = np.asarray(x, dtype=np.float64)
    x = np.maximum(x, 0.0)  # CDF is 0 for x < 0
    return 1.0 - np.exp(-np.power(x / LAMBDA_WEIBULL, K_WEIBULL))

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
    
    Strategy: Score based on the negative of "effective capacity" where
    effective capacity = remaining_space - expected_next_item_size,
    but weighted by the probability that a future item will fit.
    
    score = -(remaining - E[X]) * P(X <= remaining)
    
    This rewards bins where:
    - The remaining space is close to the expected next item size (low |remaining - E[X]|)
    - The probability of a future item fitting is high (high P(X <= remaining))
    
    Bins with very small remaining get heavily penalized (low probability).
    Bins with large remaining are scored based on deviation from mean, weighted by high probability.
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
    
    # Compute probability that a future item fits in the remaining space
    prob_fit = _weibull_cdf(remaining)
    
    # Score: negative of (remaining - expected_next) weighted by probability of fit
    # This encourages bins where remaining is close to expected_next AND prob_fit is high
    scores[valid_mask] = -(remaining - expected_next) * prob_fit
    
    return scores
# EVOLVE-BLOCK-END
