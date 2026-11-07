import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = _start_time + 300.0

def acquisition(mu, sigma, f_best):
    """
    Kriging Believer acquisition function.
    
    Instead of sampling from the posterior (Thompson Sampling) or computing
    Expected Improvement, we construct a deterministic surrogate objective:
        surrogate(x) = mu(x) - k * sigma(x)
    
    This is the Lower Confidence Bound (LCB). By evaluating the "believer"
    on this pessimistic estimate, we effectively combine exploitation
    (low mu) with exploration (high sigma) in a deterministic manner.
    
    The score returned is the negative of this surrogate value, so that
    higher scores correspond to lower predicted function values.
    
    We use a dynamic exploration coefficient k that starts high and decays
    over the course of the optimization, encouraging early exploration
    and later exploitation. However, since we don't have access to the
    step count directly, we use a fixed k that balances the two.
    
    For ST1D on [-5, 5], the function range is roughly [-39, 39].
    A reasonable k is around 1.0-2.0. We use k=1.5 as a starting point.
    """
    # Deadline guard
    if time.time() > _DEADLINE - 10.0:
        # Fallback to simple exploitation: return -mu
        return -mu
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Kriging Believer: Lower Confidence Bound
    # k controls exploration-exploitation tradeoff
    # Higher k -> more exploration
    k = 1.5
    
    # Surrogate objective: mu - k * sigma
    surrogate = mu - k * sigma_safe
    
    # Score is negative surrogate (higher score = better candidate)
    scores = -surrogate
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
