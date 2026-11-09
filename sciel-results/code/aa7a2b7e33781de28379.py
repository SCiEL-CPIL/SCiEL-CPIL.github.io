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
    
    # Calculate z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Clip EI to finite values
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Lower Confidence Bound (LCB) component for minimization
    # We want to minimize mu - beta * sigma, so we maximize -mu + beta * sigma
    # Let's use beta = 2.0
    beta = 2.0
    lcb_score = -mu + beta * sigma_safe
    
    # Normalize LCB score to be comparable to EI?
    # EI is typically small (improvement). LCB can be large.
    # Let's try a hybrid: 0.5 * EI + 0.5 * LCB_score
    # This balances exploitation (EI) and exploration (LCB)
    
    hybrid_score = 0.5 * ei + 0.5 * lcb_score
    
    # Ensure the result is finite
    hybrid_score = np.nan_to_num(hybrid_score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return hybrid_score
