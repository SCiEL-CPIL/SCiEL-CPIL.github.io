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
    
    elapsed = time.time() - _start_time
    if elapsed > 290.0:
        # Fallback: exploit
        mu = np.asarray(mu, dtype=float)
        mu = np.where(np.isfinite(mu), mu, f_best + 1e6)
        return -mu

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Sanitize
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best + 1e6)
    
    # Adaptive Beta
    t = _call_count
    beta = 1.0 + 10.0 * np.log(1.0 + t)
    
    # UCB Score
    ucb = mu + np.sqrt(beta) * sigma
    
    # Optional: Add a bit of EI to prevent pure exploration if mu is very promising
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Combine
    # Weighting: Early on, UCB dominates. Later, EI might help refine.
    # But UCB already handles this via beta.
    # Let's try a mix: 0.8 * UCB + 0.2 * EI?
    # Or just UCB. The inspiration said UCB outperforms pure EI.
    # Let's stick to UCB to be safe, or a slight blend.
    # Let's try 1.0 * UCB.
    
    score = ucb
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
