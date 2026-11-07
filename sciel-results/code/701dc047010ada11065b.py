"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global start time to enforce deadline
_start_time = time.time()

def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() - _start_time > 300:
        # Fallback: return something reasonable, e.g., UCB with beta=0 (pure exploitation)
        # Or just return -mu
        return -mu

    # Probability of Improvement (PI) for minimization
    # PI(x) = P(f(x) < f_best)
    # where f(x) ~ N(mu(x), sigma^2(x))
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    # Z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # PI = Phi(z)
    # where Phi is the CDF of the standard normal distribution
    
    # Compute Phi(z)
    pi = norm.cdf(z)
    
    # Ensure finite output
    scores = np.where(np.isfinite(pi), pi, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
