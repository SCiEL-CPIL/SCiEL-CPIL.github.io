"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for Bayesian Optimization.
    
    EI balances exploration and exploitation by estimating the expected reduction
    in the best observed value. It is generally robust for multimodal functions
    like Hartmann 6D when the Gaussian Process posterior is well-calibrated.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Mean of the GP posterior at candidate points
    sigma : array of shape (2000,)
        Standard deviation of the GP posterior at candidate points
    f_best : float
        Best (lowest) observed function value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores; higher means evaluate next
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle edge cases for invalid inputs
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        return np.zeros(len(mu))
        
    # Ensure sigma is positive to avoid division by zero
    # If sigma is very small, the posterior is certain.
    sigma = np.maximum(sigma, 1e-10)
    
    # Calculate the Z-score: (f_best - mu) / sigma
    # This represents how many standard deviations f_best is above the mean.
    z = (f_best - mu) / sigma
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the CDF and phi is the PDF of the standard normal distribution.
    
    # For numerical stability, we can use the property that for z < -26, Phi(z) is 0.
    # However, scipy's norm handles this well.
    
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    scores = (f_best - mu) * cdf + sigma * pdf
    
    # Ensure scores are finite and non-negative (EI should be >= 0)
    scores = np.maximum(scores, 0.0)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
