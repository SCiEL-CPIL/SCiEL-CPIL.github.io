import numpy as np
from scipy.stats import norm
import time

_start_time = None
DEADLINE = 300.0
SAFETY = 20.0

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.perf_counter()
    return time.perf_counter() - _start_time

def _ei(mu, s, f_best, xi=0.0):
    """Expected Improvement for minimization."""
    z = (f_best - mu - xi) / s
    return (f_best - mu - xi) * norm.cdf(z) + s * norm.pdf(z)

def acquisition(mu, sigma, f_best):
    global _start_time
    if _start_time is None:
        _start_time = time.perf_counter()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = mu.shape[0]
    
    # Ensure sigma is strictly positive
    s = np.where(sigma > 1e-9, sigma, 1e-9)
    
    # Sanitize f_best
    fb = float(f_best)
    if not np.isfinite(fb):
        fb = 0.0
    
    # Deadline guard
    if _elapsed() > DEADLINE - SAFETY:
        # Fallback: Pure Exploitation (argmin mu)
        return -mu
    
    # Compute EI scores
    ei = _ei(mu, s, fb)
    
    # Add a small exploration bonus that rewards uncertainty
    # This helps escape local optima in the Weierstrass landscape
    # The bonus is proportional to sigma and the probability of being a new best
    p_new_best = norm.cdf((fb - mu) / s)
    bonus = 0.1 * s * p_new_best
    
    scores = -ei + bonus
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
