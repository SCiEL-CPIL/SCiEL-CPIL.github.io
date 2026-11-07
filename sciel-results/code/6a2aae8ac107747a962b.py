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
        Expected Improvement scores. Higher values indicate higher priority.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-12)
    
    # Standardized improvement
    # We want to minimize f, so improvement is (f_best - mu)
    # Z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # EI formula:
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Where Phi is CDF and phi is PDF of standard normal
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    scores = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e308, neginf=0.0)
    
    return scores
