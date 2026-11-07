"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for minimization.
    
    For each candidate x, the posterior is N(mu, sigma^2).
    The improvement is defined as I(x) = max(0, f_best - f(x)).
    The expected improvement is:
    EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma.
    
    This approach typically outperforms UCB in low-dimensional smooth problems
    like the 1D sphere function because it weights candidates by the *magnitude*
    of potential improvement rather than just the confidence interval width.
    """
    # Ensure sigma is strictly positive to avoid division by zero in z calculation
    # A very small epsilon is used to handle points with near-zero uncertainty.
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    # Phi(z) is the CDF of the standard normal distribution
    # phi(z) is the PDF of the standard normal distribution
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure finite scores
    ei = np.where(np.isfinite(ei), ei, -np.inf)
    
    return ei
# EVOLVE-BLOCK-END
