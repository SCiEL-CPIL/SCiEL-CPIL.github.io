"""
Acquisition function for Bayesian optimization of Weierstrass 1D.
Uses Expected Improvement (EI) for minimization.
"""
import numpy as np
from scipy.stats import norm
import time

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 300  # seconds

def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() - START_TIME > TIME_LIMIT - 5:
        # Fallback: return LCB with a conservative beta
        return mu - 2.0 * sigma

    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Calculate z-score
    # z = (f_best - mu) / sigma
    # For minimization, we want mu < f_best.
    # If mu < f_best, z is positive.
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement (PI) for minimization:
    # P(f(x) < f_best) = Phi(z)
    # This acquisition function favors points with a high probability
    # of being better than the current best. It is more exploitative
    # than EI and can be better for rugged landscapes by focusing
    # on promising basins rather than maximizing expected gain.
    
    pi = norm.cdf(z)
    
    # Clip to finite values
    pi = np.nan_to_num(pi, nan=0.0, posinf=1.0, neginf=0.0)
    
    return pi
