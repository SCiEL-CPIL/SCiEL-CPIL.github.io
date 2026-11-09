import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Computes the Expected Improvement (EI) acquisition function for minimization.
    
    Parameters:
    mu : array-like, shape (N,)
        Posterior mean at N candidate points.
    sigma : array-like, shape (N,)
        Posterior standard deviation at N candidate points.
    f_best : float
        Current best observed function value.
        
    Returns:
    scores : ndarray, shape (N,)
        Acquisition scores for the candidate points (higher is better).
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Calculate improvement: how much better than f_best
    improvement = f_best - mu
    
    # Expected Improvement (EI) for minimization
    # EI = E[max(f_best - f, 0)]
    # For Gaussian: EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    from scipy.stats import norm
    
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(ei), ei, 0.0)
    
    return scores
