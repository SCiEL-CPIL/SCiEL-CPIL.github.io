import numpy as np
import time
import scipy.special

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 135.0  # Safety margin before 150s kill

# Weibull distribution parameters for the generated sequences
# Assuming a standard Weibull distribution with shape k=2.0 and scale lambda=25.0
# This yields a mean of ~22 and a median of ~19, which fits well into capacity 100.
# For Weibull(k, lambda), E[X] = lambda * Gamma(1 + 1/k)
# Median = lambda * (ln(2))^(1/k)
k = 2.0
lambda_scale = 25.0
expected_next_item = lambda_scale * scipy.special.gamma(1 + 1.0 / k)
target_remainder = lambda_scale * (np.log(2) ** (1.0 / k))  # 50th percentile (median)

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Deadline guard: if we're too close to the limit, return a safe default
    if time.time() - _START_TIME > _DEADLINE:
        valid = bins >= item
        if not np.any(valid):
            return np.full_like(bins, -np.inf)
        scores = -(bins - item)
        scores[~valid] = -np.inf
        return scores

    # Determine which bins have enough remaining capacity for the current item
    valid = bins >= item
    
    if not np.any(valid):
        return np.full_like(bins, -np.inf)
        
    # Strategy: Discrete-time control system steering toward stable equilibrium
    # We model the bin packing process as a control system where the score for each bin
    # is the negative of the squared deviation between the resulting remainder and the
    # target remainder derived from the inverse of the Weibull CDF at the 50th percentile.
    
    # The resulting remainder after placing the item is:
    gap = bins - item
    
    # Target remainder is the median of the Weibull distribution:
    # target_remainder = lambda_scale * (ln(2))^(1/k)
    
    # Score = -(gap - target_remainder)^2
    # This steers the system toward a statistically stable equilibrium of gap sizes.
    
    deviation = gap - target_remainder
    scores = -deviation ** 2
    
    # Mask invalid bins
    scores[~valid] = -np.inf
    
    return scores
