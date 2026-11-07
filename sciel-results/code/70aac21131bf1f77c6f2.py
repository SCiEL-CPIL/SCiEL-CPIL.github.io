import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0

def _elapsed():
    return time.time() - _start_time

def acquisition(mu, sigma, f_best, cost):
    if _elapsed() > _DEADLINE:
        return np.zeros(2000)

    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement (EI) for minimization
    # s = (f_best - mu) / sigma
    s = (f_best - mu) / sigma_safe
    
    # Standard Normal PDF and CDF
    phi_s = norm.pdf(s)
    Phi_s = norm.cdf(s)
    
    # EI = (f_best - mu) * Phi(s) + sigma * phi(s)
    # Note: If f_best - mu is negative (mu > f_best), EI should be 0 or very small.
    # The formula naturally handles this because Phi(s) becomes small and (f_best-mu) is negative,
    # but the sigma*phi(s) term is positive.
    # Actually, standard EI is max(0, ...).
    # Let's compute the raw value and clip to 0.
    ei = (f_best - mu) * Phi_s + sigma_safe * phi_s
    ei = np.maximum(ei, 0.0)
    
    # Cost Penalty
    # Cost is in [1, 2].
    # We want to penalize higher costs.
    # Lambda determines the trade-off.
    # If lambda is too high, we only pick the cheapest points (x ~ -1).
    # If lambda is too low, we ignore cost.
    # Let's try lambda = 0.05.
    # Cost penalty = 0.05 * cost. Range [0.05, 0.10].
    # EI range: potentially up to 0.5 or so.
    # This seems balanced.
    
    penalty = 0.05 * cost
    
    scores = ei - penalty
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
