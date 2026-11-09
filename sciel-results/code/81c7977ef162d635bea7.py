# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) with a small exploitation bias.
    This subtracts a small multiple of the standard deviation to prefer regions
    with high expected improvement and low uncertainty, countering over-exploration.
    """
    # Handle near-zero sigma to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement (for minimization)
    z = (f_best - mu) / sigma_safe
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Exploitation bias: subtract a small multiple of sigma
    # This makes the acquisition function more conservative than pure EI
    lambda_exploit = 0.1
    score = ei - lambda_exploit * sigma
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
