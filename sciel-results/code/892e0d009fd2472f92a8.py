"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Compute a cost-aware acquisition score using Expected Improvement (EI).
    
    For minimization, EI measures the expected amount by which the function
    value at a point might be lower than the current best.
    Score = EI / cost.
    
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
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate the improvement: f_best - mu
    # If mu < f_best, this is positive, indicating potential improvement.
    improvement = f_best - mu
    
    # Standardized improvement z
    z = improvement / sigma_safe
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the CDF and phi is the PDF of the standard normal distribution
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = improvement * Phi_z + sigma_safe * phi_z
    
    # Numerical stability: clip negative values (should be ~0)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = ei / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
