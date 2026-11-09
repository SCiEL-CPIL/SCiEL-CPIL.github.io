"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    
    Parameters
    ----------
    mu : array of shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : array of shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns
    -------
    array of shape (2000,)
        Acquisition scores; higher values indicate more promising points.
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Handle edge cases where z is very large or very small
    # For very negative z, EI approaches 0
    # For very positive z, EI approaches (f_best - mu)
    
    # Expected Improvement formula
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the standard normal CDF and phi is the standard normal PDF
    
    # To handle numerical issues, we compute this carefully
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # EI for minimization: we want to find points where f_best - mu is large (mu << f_best)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # For points where mu > f_best, EI should be 0 (no improvement expected)
    # This is naturally handled by the formula, but let's ensure non-negativity
    ei = np.maximum(ei, 0.0)
    
    # Add a small exploration bonus based on uncertainty
    # This helps when we have little data
    # The uncertainty bonus is proportional to sigma, but we scale it down
    # as we get closer to the optimum
    uncertainty_bonus = 0.1 * sigma_safe
    
    # Combine EI with uncertainty bonus
    score = ei + uncertainty_bonus
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
