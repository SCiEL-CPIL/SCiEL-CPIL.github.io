"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import weibull_min
from scipy.special import gamma

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

# Precompute mean item size
_MEAN_ITEM = _WEIBULL_LAMBDA * gamma(1.0 + 1.0 / _WEIBULL_K)

def _weibull_cdf(x):
    """Compute Weibull CDF for array x."""
    x = np.maximum(x, 0.0)
    return 1.0 - np.exp(-(x / _WEIBULL_LAMBDA) ** _WEIBULL_K)

def _expected_wasted_space(remaining):
    """
    Estimate the expected wasted space in a bin with 'remaining' capacity.
    
    The expected wasted space is: remaining - E[min(X, remaining)]
    where X ~ Weibull(k, lambda).
    
    E[min(X, remaining)] = integral_0^remaining P(X > t) dt
                         = integral_0^remaining (1 - CDF(t)) dt
                         = integral_0^remaining exp(-(t/lambda)^k) dt
    
    For Weibull, this integral can be approximated. For small remaining,
    most of the mass is below remaining, so E[min(X, remaining)] ≈ remaining.
    For large remaining, E[min(X, remaining)] ≈ E[X] = mean_item.
    
    We use a smooth interpolation based on the CDF value.
    """
    # If remaining is very small, almost all items will be larger, so we waste most of it
    # If remaining is very large, we'll fill it well, so we waste little
    
    # Use the survival function S(t) = P(X > t) = exp(-(t/lambda)^k)
    # E[min(X, r)] = integral_0^r S(t) dt
    
    # Approximate the integral using the fact that for Weibull:
    # When r << lambda, S(t) ≈ 1 - (t/lambda)^k, so integral ≈ r - r^(k+1)/((k+1)*lambda^k)
    # When r >> lambda, S(t) decays rapidly, so integral ≈ lambda * gamma(1 + 1/k) = mean_item
    
    # A good approximation: use the harmonic mean between r and mean_item,
    # weighted by the CDF at r
    cdf_r = _weibull_cdf(remaining)
    
    # If cdf_r is close to 1, remaining is large enough that most items fit,
    # so we can fill it well: wasted ≈ 0
    # If cdf_r is close to 0, remaining is small, few items fit: wasted ≈ remaining
    
    # Better: expected fill = E[min(X, r)]
    # Approximate as: fill = r * cdf_r + mean_item * (1 - cdf_r) * cdf_r
    # This is not quite right. Let's use:
    # fill ≈ r * S(r) + integral_0^r S(t)dt approximation
    
    # Simpler robust approach: 
    # For small r: E[min(X,r)] ≈ r * (1 - (r/lambda)^k / (k+1)) ≈ r * (1 - r^k/(lambda^k*(k+1)))
    # For large r: E[min(X,r)] ≈ mean_item
    
    # Use a blend:
    ratio = remaining / _WEIBULL_LAMBDA
    # For ratio < 1, use Taylor expansion approximation
    # For ratio > 1, use mean_item with correction
    
    # Approximation: E[min(X,r)] = r - r * (1 - cdf_r) * (1 + 1/k) * (r/lambda)^k / (k+1)
    # This is getting complicated. Let's use a simpler but effective heuristic.
    
    # Heuristic: expected wasted = remaining * (1 - cdf_r) * (1 + remaining / (2 * _MEAN_ITEM))
    # The (1 + remaining/(2*mean)) factor accounts for the fact that even when items fit,
    # there's some overfill waste on average.
    
    wasted = remaining * (1.0 - cdf_r) * (1.0 + 0.5 * remaining / (_MEAN_ITEM + 1e-10))
    
    # Clip to reasonable bounds
    wasted = np.clip(wasted, 0.0, remaining)
    
    return wasted

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Scores each bin. The item goes to the bin with the highest score.
    bins: array of remaining capacities for each bin.
    item: the size of the item to place.
    
    Strategy: We want to minimize the expected total wasted space across all bins.
    When placing an item in a bin, we reduce its remaining capacity.
    The score should reflect how "good" it is to leave the remaining space in that bin.
    
    We use a combination of:
    1. Best Fit: prefer bins with less remaining (tighter packing)
    2. Wasted space penalty: penalize remaining space that is unlikely to be filled
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
    
    # Compute expected wasted space for each valid remaining
    wasted = _expected_wasted_space(valid_remaining)
    
    # Score: we want to minimize (remaining + wasted)
    # But remaining already includes the waste if no item fits.
    # Actually, the total "cost" of a bin state is the expected wasted space.
    # When we place an item, the new remaining is 'valid_remaining'.
    # The expected wasted space in that new state is 'wasted'.
    # So we should minimize 'wasted' primarily.
    # But also, having less remaining means we're using fewer bins, which is good.
    # 
    # Combine: score = -(wasted + alpha * remaining)
    # where alpha is a small weight to prefer tighter packing when waste is similar.
    
    alpha = 0.1  # small weight on remaining space
    
    scores_valid = -(wasted + alpha * valid_remaining)
    
    scores[valid_mask] = scores_valid
    
    return scores
# EVOLVE-BLOCK-END
