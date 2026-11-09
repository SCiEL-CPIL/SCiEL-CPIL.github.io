"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    
    Uses Expected Improvement (EI) for minimization.
    This is a standard, robust acquisition function that balances
    exploration and exploitation by maximizing the expected reduction
    in the best observed value.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : np.ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    np.ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Clip sigma to avoid division by zero and numerical issues
    # We use a small epsilon to prevent inf/nan
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate the Z-score for Expected Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the CDF and phi is the PDF of the standard normal distribution
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure finite scores. EI should naturally be non-negative and finite
    # if inputs are well-behaved, but we guard against edge cases.
    scores = np.where(np.isfinite(ei), ei, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
