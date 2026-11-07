"""Rewritten acquisition function using Expected Improvement (EI)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6D.
    
    Uses Expected Improvement (EI) to balance exploration and exploitation.
    EI measures the expected reduction in the objective value relative to the current best f_best.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at candidate points (in objective units, lower is better).
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Lowest observed objective value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising to evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate the improvement term: f_best - mu
    # Since we are minimizing, if mu < f_best, the improvement is positive.
    improvement = f_best - mu
    
    # Calculate the z-score for the standard normal distribution
    z = improvement / sigma
    
    # Expected Improvement formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the CDF and phi is the PDF of the standard normal distribution
    
    # Use scipy.stats.norm for numerical stability
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    scores = improvement * Phi_z + sigma * phi_z
    
    # Ensure all scores are finite and non-negative (EI is always >= 0)
    scores = np.maximum(0.0, scores)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
