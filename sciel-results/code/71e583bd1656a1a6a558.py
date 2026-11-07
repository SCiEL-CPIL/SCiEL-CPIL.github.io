import numpy as np
import time
from scipy.stats import norm

_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best):
    global _start_time, _call_count
    if _start_time is None:
        _start_time = time.time()
    
    _call_count += 1
    
    # Deadline guard
    if time.time() - _start_time > 325:
        # Fallback to EI if time is tight
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.where(np.isfinite(ei), ei, 0.0)
        return ei

    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Structural Change: Gaussian-relevance weighted uncertainty metric
    # This balances exploration (sigma) with the probability that the point
    # is actually near the current best (via the Gaussian weight on z).
    
    # z-score: how many sigmas below f_best is the mean?
    # If mu < f_best, z is positive.
    z = (f_best - mu) / sigma_safe
    
    # Gaussian weight: peaks when z is small (mu close to f_best)
    # This downweights regions where mu is much worse than f_best (large negative z)
    # and also downweights regions where mu is much better (large positive z) 
    # although the -mu term handles the latter.
    # Actually, we want to explore where there is uncertainty AND potential.
    # The term exp(-0.5 * z^2) is the PDF of standard normal.
    
    # Decay beta from high exploration to low exploration
    # 25 optimization steps
    progress = min(_call_count / 25.0, 1.0)
    # Start with high exploration, end with exploitation
    beta = 4.0 - 3.0 * progress
    
    # Score: Exploit (-mu) + Explore (beta * sigma * relevance)
    # Relevance is high when the distribution overlaps with f_best
    relevance = np.exp(-0.5 * z**2)
    
    score = -mu + beta * sigma_safe * relevance
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
