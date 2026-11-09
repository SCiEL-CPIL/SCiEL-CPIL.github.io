import numpy as np
from scipy.stats import norm
import time

_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    if elapsed > 320:
        # Fallback: Greedy exploitation
        scores = -mu
        return np.where(np.isfinite(scores), scores, 0.0)

    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Expected Improvement for minimization
    # f_best is the current best (minimum) value.
    # We want to maximize the expected reduction in f.
    improvement = f_best - mu
    Z = improvement / sigma_safe
    
    # EI = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    ei = improvement * norm.cdf(Z) + sigma_safe * norm.pdf(Z)
    
    scores = np.where(np.isfinite(ei), ei, 0.0)
    return scores
