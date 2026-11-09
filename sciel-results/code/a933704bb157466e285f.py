"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    Uses standard Expected Improvement (EI).
    """
    # Use a small epsilon to prevent division by zero if sigma is 0
    sigma_safe = sigma + 1e-9
    
    # Standard Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure all scores are finite as required by the contract
    scores = np.where(np.isfinite(ei), ei, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
