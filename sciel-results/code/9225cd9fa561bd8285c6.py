"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Compute a cost-aware acquisition score using Expected Improvement (EI).
    
    For minimization, EI is the expected amount by which a new evaluation 
    might improve upon the current best f_best.
    We divide the EI by the evaluation cost to get a cost-effective score.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean predictions from GP
    sigma : array-like, shape (2000,)
        Standard deviation predictions from GP
    f_best : float
        Best observed value so far
    cost : array-like, shape (2000,)
        Evaluation costs for each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate the improvement d = f_best - mu
    # We are interested in points where mu < f_best, i.e., d > 0
    d = f_best - mu
    
    # Initialize EI to 0
    ei = np.zeros_like(mu)
    
    # Only calculate EI for points where there is potential improvement (d > 0)
    mask = d > 0
    
    if np.any(mask):
        d_masked = d[mask]
        sigma_masked = sigma_safe[mask]
        
        # Standardize the improvement
        z = d_masked / sigma_masked
        
        # EI formula for minimization:
        # EI = d * Phi(z) + sigma * phi(z)
        # Phi is CDF, phi is PDF of standard normal
        phi_z = norm.pdf(z)
        Phi_z = norm.cdf(z)
        
        ei_masked = d_masked * Phi_z + sigma_masked * phi_z
        
        ei[mask] = ei_masked
    
    # Cost-aware score
    # Avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = ei / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
