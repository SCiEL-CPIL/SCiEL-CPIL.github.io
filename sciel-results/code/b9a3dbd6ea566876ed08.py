# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function using Cost-Aware Expected Improvement (CA-EI).
    
    This approach calculates the expected improvement over the current best
    value and normalizes it by the evaluation cost. This is distinct from
    linear heuristics like LCB/UCB as it uses the full Gaussian distribution
    to estimate the expected gain.
    
    The Expected Improvement (EI) is defined as:
    EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma, Phi is the standard normal CDF, and
    phi is the standard normal PDF.
    
    The score is:
    score = EI / cost
    
    This encourages points with high expected improvement relative to their
    cost.
    """
    # Ensure sigma is non-negative and stable
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement
    # Use scipy.stats.norm for Phi (cdf) and phi (pdf)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical errors can cause small negatives)
    ei = np.maximum(ei, 0.0)
    
    # Denominator: Cost
    cost_safe = np.maximum(cost, 1e-8)
    
    # Score: EI divided by cost
    score = ei / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
