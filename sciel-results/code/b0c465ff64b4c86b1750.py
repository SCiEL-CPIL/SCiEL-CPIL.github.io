import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function for Bayesian optimization.
    
    This is a deterministic, principled acquisition function that balances exploration
    and exploitation based on the expected improvement over the current best f_best.
    Higher EI means higher priority for evaluation.
    
    Parameters:
    -----------
    mu : array-like, shape (n_candidates,)
        Mean of the GP posterior at the candidate points.
    sigma : array-like, shape (n_candidates,)
        Standard deviation of the GP posterior at the candidate points.
    f_best : float
        The best observed value so far.
        
    Returns:
    --------
    scores : ndarray, shape (n_candidates,)
        The EI scores. Higher values indicate higher priority for evaluation.
    """
    # Ensure sigma is strictly positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Standardized distance to the current best
    z = (f_best - mu) / sigma
    
    # Expected Improvement formula:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the CDF of the standard normal distribution
    # and phi is the PDF of the standard normal distribution
    
    scores = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
