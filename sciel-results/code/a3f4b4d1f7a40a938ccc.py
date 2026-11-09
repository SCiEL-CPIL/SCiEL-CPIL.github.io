"""Rewritten program using Cost-Weighted Expected Improvement (CW-EI)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Weighted Expected Improvement (CW-EI) acquisition function.
    
    This approach uses Expected Improvement (EI) as the base acquisition function,
    which is known to be robust and does not require tuning a beta parameter.
    The EI score is then divided by the evaluation cost to favor cheaper points
    that offer a good expected improvement relative to their expense.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior.
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best observed value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : array, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    z = (f_best - mu) / sigma_safe
    
    # Compute Phi(z) and phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI calculation
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Clip negative EI to 0 (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-weighted score
    # We divide EI by cost to prefer cheaper evaluations
    scores = ei / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
