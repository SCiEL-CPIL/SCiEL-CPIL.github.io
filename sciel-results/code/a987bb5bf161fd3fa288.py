import numpy as np
import time

# Global state for deadline guard
_START_TIME = time.time()
_DEADLINE = 145.0  # 5 seconds margin before 150s limit

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity for the arriving item.
    The item goes to the highest-scoring bin.
    
    Strategy:
    - Model the bin packing process as a discrete-time inventory system.
    - The score for a bin is the negative of the squared deviation between the resulting remainder
      and the mode of the Weibull item size distribution.
    - This steers remainders toward the most probable future item size to maximize the likelihood
      of a perfect fit and minimize fragmentation.
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
    
    # Weibull parameters (assumed based on typical distributions in this task)
    # Shape parameter k and scale parameter lambda
    k = 0.8
    lam = 20.0
    
    # The mode of a Weibull distribution is:
    # mode = lambda * ((k-1)/k)^(1/k) for k > 1
    # For k < 1, the mode is 0 (the PDF is monotonically decreasing)
    # For k = 1 (exponential), the mode is 0
    # For k < 1, the most probable item size is near 0, but the density is highest at 0.
    # However, in bin packing, we want remainders that can fit *some* future item.
    # If k < 1, the distribution is heavy-tailed with high probability near 0.
    # Let's use the mean as a proxy for the "typical" item size when k <= 1.
    # Mean = lambda * gamma(1 + 1/k)
    
    from scipy import special
    
    if k > 1:
        mode = lam * ((k - 1) / k) ** (1.0 / k)
    else:
        # For k <= 1, the mode is 0, but using 0 as a target remainder is not useful.
        # Instead, use the mean as a representative "typical" item size.
        mean = lam * special.gamma(1.0 + 1.0 / k)
        mode = mean
    
    # The target remainder is the mode of the Weibull distribution.
    # We want to minimize the squared deviation between the remainder and the mode.
    # Score = -(remainder - mode)^2
    
    r = valid_remaining
    deviation = r - mode
    score = -deviation ** 2
    
    # Assign scores back to the original array
    scores[valid_mask] = score
    
    return scores
