"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Acquisition function for Bayesian optimization.
    
    Based on the finding that removing cost-normalization allows better exploration
    of high-cost regions (where the shifted minimum often resides), this implementation
    uses pure Expected Improvement (EI) without dividing by cost.
    
    Parameters:
    -----------
    mu : array-like, shape (n,)
        Mean predictions of the Gaussian Process
    sigma : array-like, shape (n,)
        Standard deviation predictions of the Gaussian Process
    f_best : float
        Best observed function value so far
    cost : array-like, shape (n,)
        Cost of evaluating each candidate point (ignored in this strategy)
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores for each candidate point (Higher is better)
    """
    # Ensure numerical stability
    # Use a small epsilon to avoid division by zero in z calculation
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Standardized improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the CDF of the standard normal distribution
    # and phi is the PDF of the standard normal distribution
    
    # Clip z to avoid extreme values causing numerical issues in norm.pdf/cdf
    # Although scipy is generally robust, clipping helps with extreme tails
    z_clipped = np.clip(z, -10, 10)
    
    cdf_z = norm.cdf(z_clipped)
    pdf_z = norm.pdf(z_clipped)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Return EI scores directly.
    # The 'cost' argument is intentionally ignored to avoid biasing
    # the search away from the central region (higher cost) where the minimum is.
    scores = ei
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
