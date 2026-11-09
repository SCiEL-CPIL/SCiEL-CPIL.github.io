import numpy as np
from scipy.stats import norm
import time

_start_time = None

def acquisition(mu, sigma, f_best):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    # If we're close to the deadline, return a safe fallback
    if elapsed > 320:
        # Return UCB-like scores as fallback
        return -mu + 2.0 * sigma
    
    # Expected Improvement for minimization
    # EI(x) = E[max(0, f_best - f(x))]
    # where f(x) ~ N(mu, sigma^2)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # z = (f_best - mu) / sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add a small exploration bonus to encourage exploration early
    # This helps when f_best is still far from the true minimum
    # The bonus decays as sigma decreases (more certain predictions)
    exploration_bonus = 0.1 * sigma_safe
    scores = ei + exploration_bonus
    
    return scores
