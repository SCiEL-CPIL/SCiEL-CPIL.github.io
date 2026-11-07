"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    # Safety guards
    if len(mu) != len(sigma):
        return np.zeros(len(mu))
    
    # Track elapsed time for safety
    t_start = time.time()
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Calculate z-score for Probability of Improvement (PI)
    # z = (f_best - mu) / sigma
    # Higher z means higher probability that y < f_best
    z = (f_best - mu) / sigma_safe
    
    # The acquisition function is the Probability of Improvement (PI)
    # PI = Phi(z) where Phi is the standard normal CDF
    # This is a deterministic, stable strategy that prioritizes points
    # with a high likelihood of improving upon f_best, naturally balancing
    # exploration (high sigma -> moderate z if mu is not too bad) and
    # exploitation (low mu -> high z).
    
    scores = norm.cdf(z)
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Safety deadline guard: if this took too long, return what we have.
    # The computation is vectorized and fast, so this should never trigger,
    # but it's required by the directive.
    elapsed = time.time() - t_start
    if elapsed > 290:
        # If we are close to the limit, just return the current scores.
        # There is no additional search to abort.
        pass
        
    return scores
# EVOLVE-BLOCK-END
