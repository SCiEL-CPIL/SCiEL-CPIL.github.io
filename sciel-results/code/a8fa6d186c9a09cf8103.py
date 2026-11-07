"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_call_count = 0

def acquisition(mu, sigma, f_best):
    global _call_count
    _call_count += 1
    
    # Deadline guard: if we are too close to the limit, return a simple score
    # to ensure we don't timeout.
    if time.time() - _start_time > 325:
        # Fallback to a simple greedy score
        return -mu 
    
    # t is the number of optimization steps taken (excluding initial 5)
    # _call_count is 1 for the first optimization step
    t = _call_count
    
    # Handle case where sigma is very small or zero to avoid division by zero
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute Standard EI
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Time-decaying exploration bonus
    # We add a small multiple of sigma to encourage exploration in early steps.
    # w = 0.5 / (1 + t) 
    # At t=1, w=0.25. At t=5, w=0.08.
    # This makes the score slightly more exploratory than pure EI early on.
    w = 0.5 / (1.0 + t)
    
    # The bonus is w * sigma. 
    # Note: EI already scales with sigma in the second term (sigma * pdf(z)).
    # pdf(z) is max 0.4. So EI's sigma term is at most 0.4 * sigma.
    # Our bonus is 0.25 * sigma at start. This is significant.
    
    scores = ei + w * sigma_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
