# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global timer for safety
_t0 = time.time()
_DEADLINE = 320.0

def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() - _t0 > _DEADLINE:
        return -mu

    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Use Probability of Improvement (PI) acquisition function
    # PI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # This prioritizes points with high probability of beating the current best
    # while naturally dampening pure noise exploration in the low-budget regime
    
    # Compute the normalized improvement
    improvement = (f_best - mu) / sigma_safe
    
    # Probability of improvement: CDF of standard normal
    pi_score = norm.cdf(improvement)
    
    # Ensure finite scores
    pi_score = np.where(np.isfinite(pi_score), pi_score, 0.0)
    
    return pi_score
# EVOLVE-BLOCK-END
