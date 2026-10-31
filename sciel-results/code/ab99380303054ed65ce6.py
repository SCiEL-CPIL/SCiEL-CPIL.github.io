"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import weibull_min

# Global state for deadline guard
_START_TIME = time.time()
_DEADLINE = 145.0  # 5 seconds margin before 150s

# Precompute Weibull parameters for the scoring function
# Weibull distribution with shape k and scale lambda
# For capacity 100, typical Weibull parameters for bin packing:
# k (shape) around 1.5-2.5, lambda (scale) around 10-20
# We'll use a representative distribution to estimate future item sizes
_WEIBULL_K = 2.0  # shape parameter
_WEIBULL_LAMBDA = 15.0  # scale parameter
_WEIBULL_MIN = 0.0  # minimum parameter

def _weibull_cdf(x):
    """Compute Weibull CDF for array x."""
    x = np.maximum(x, 0.0)
    return 1.0 - np.exp(-(x / _WEIBULL_LAMBDA) ** _WEIBULL_K)

def _expected_regret(remaining):
    """
    Estimate the expected regret of leaving 'remaining' space.
    Regret is high if the remaining space is unlikely to be filled by future items.
    We model this as: P(no future item fits in remaining) * remaining
    Plus a penalty for wasted space if no item fits.
    """
    # Probability that a single future item does NOT fit in the remaining space
    # This is P(item > remaining) = 1 - CDF(remaining)
    p_no_fit = 1.0 - _weibull_cdf(remaining)
    
    # Expected number of items needed to fill the space if items do fit
    # Mean item size
    mean_item = _WEIBULL_LAMBDA * (1 + _WEIBULL_K) ** (-1.0 / _WEIBULL_K)
    # Actually, mean of Weibull = lambda * Gamma(1 + 1/k)
    from scipy.special import gamma
    mean_item = _WEIBULL_LAMBDA * gamma(1.0 + 1.0 / _WEIBULL_K)
    
    # If remaining < mean_item, it's harder to fill
    # Regret increases as remaining becomes too small to be useful
    # Use a smooth penalty: regret = remaining * p_no_fit + (1 - p_no_fit) * max(0, remaining - mean_item)
    # This captures: if no item fits, we waste 'remaining'. If items fit, we waste the excess over mean.
    
    # Simpler: regret = remaining * (1 - _weibull_cdf(remaining))
    # This is the expected wasted space assuming one attempt.
    # But we can try multiple items. Let's use a more refined estimate.
    
    # Expected wasted space = remaining - E[min(item, remaining)]
    # E[min(item, remaining)] = integral_0^remaining P(item > t) dt = integral_0^remaining (1 - CDF(t)) dt
    # For Weibull, this integral has a closed form but let's approximate.
    
    # Approximate E[min(item, remaining)] using the fact that for small remaining,
    # E[min(item, remaining)] ≈ remaining * CDF(remaining) / mean_item * remaining
    # Actually, let's just use: regret = remaining * (1 - CDF(remaining))
    # This is a reasonable proxy for wasted space.
    
    regret = remaining * (1.0 - _weibull_cdf(remaining))
    return regret

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin. The item goes to the bin with the highest score.
    bins: array of remaining capacities for each bin.
    item: the size of the item to place.
    """
    # Deadline check: if we are running out of time, fall back to simple Best Fit
    if time.time() > _DEADLINE:
        remaining = bins - item
        scores = np.full_like(bins, -1e9, dtype=float)
        valid = remaining >= 0
        scores[valid] = -remaining[valid]
        return scores

    # Calculate remaining capacity after placing the item
    remaining = bins - item
    
    # Mask for bins that can fit the item
    valid_mask = remaining >= 0
    
    # Initialize scores to a very low value (negative infinity) for invalid bins
    scores = np.full_like(bins, -1e9, dtype=float)
    
    # If no bin can fit the item, return all low scores
    if not np.any(valid_mask):
        return scores

    # Extract valid bins' remaining capacities
    valid_remaining = remaining[valid_mask]
    
    # Score: minimize the sum of:
    # 1. The remaining space (to pack tightly)
    # 2. The expected regret of the remaining space (to avoid slivers)
    # Score = -remaining - regret(remaining)
    
    # Compute regret for each valid remaining
    regrets = _expected_regret(valid_remaining)
    
    # Final score: lower remaining is better, but penalize high regret
    scores_valid = -(valid_remaining + regrets)
    
    scores[valid_mask] = scores_valid
    
    return scores
# EVOLVE-BLOCK-END
