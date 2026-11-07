import numpy as np

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
        EI scores. Higher values indicate higher priority for evaluation.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute the improvement: f_best - mu (how much better the candidate could be)
    improvement = f_best - mu
    
    # Standardize: z = improvement / sigma
    z = improvement / sigma
    
    # Compute EI for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the CDF of the standard normal and phi is the PDF.
    # Using scipy for numerical stability
    from scipy.stats import norm
    
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    # Handle edge cases where z is very large or small
    # For very negative z, EI should be close to 0
    # For very positive z, EI ≈ improvement
    
    ei = np.zeros_like(mu)
    
    # Where improvement is positive (candidate is promising)
    positive = improvement > 0
    ei[positive] = improvement[positive] * Phi[positive] + sigma[positive] * phi[positive]
    
    # Where improvement is zero or negative, EI is 0 (or very close to 0)
    # For numerical stability, set very small negative values to 0
    ei = np.maximum(ei, 0)
    
    # Ensure all scores are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
