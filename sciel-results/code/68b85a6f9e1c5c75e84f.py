"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Pure Expected Improvement (EI) acquisition function.
    
    EI is the standard choice for Bayesian optimization and performs well
    when the candidate pool is large (2000 uniform samples per step), as the
    diversity of candidates naturally provides exploration.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure finite output
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    return ei
# EVOLVE-BLOCK-END
