"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function with adaptive exploration.
    
    EI is a different acquisition family from LCB. It computes the expected
    improvement over the current best value, which naturally balances exploration
    and exploitation. We use an adaptive xi parameter to encourage exploration
    in the early stages.
    """
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Adaptive exploration parameter xi
    # Early on, use higher xi to encourage exploration; later, reduce it
    sigma_mean = np.mean(sigma_safe)
    sigma_max = np.max(sigma_safe)
    
    if sigma_max > 1e-10:
        conv_ratio = sigma_mean / sigma_max
    else:
        conv_ratio = 1.0
    
    # xi starts higher and decreases as we converge
    xi_base = 0.1
    xi = xi_base * (0.5 + 0.5 * conv_ratio)
    
    # Compute Expected Improvement
    # EI = (f_best - mu) * Phi((f_best - mu) / sigma) + sigma * phi((f_best - mu) / sigma)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Subtract xi to encourage exploration (higher xi = more exploration)
    ei_scores = ei - xi
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(ei_scores), ei_scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
