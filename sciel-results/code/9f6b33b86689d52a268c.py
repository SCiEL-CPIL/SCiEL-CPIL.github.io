import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI).
    
    Parameters:
    mu : array-like, shape (N,)
        Mean of the GP posterior.
    sigma : array-like, shape (N,)
        Standard deviation of the GP posterior.
    f_best : float
        Best observed function value so far.
    cost : array-like, shape (N,)
        Cost of evaluating each candidate point.
        
    Returns:
    scores : array, shape (N,)
        Acquisition scores (higher is better).
    """
    # Avoid division by zero and numerical instability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate improvement
    imp = f_best - mu
    
    # Calculate z-score
    z = imp / sigma_safe
    
    # Calculate EI
    # EI = imp * Phi(z) + sigma * phi(z)
    ei = imp * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Handle cases where EI might be negative due to numerical issues or if mu > f_best significantly
    # EI should be non-negative. If imp < 0, EI is small.
    # We clip to 0 to be safe, as negative EI implies no expected improvement.
    ei = np.maximum(ei, 0.0)
    
    # Cost-adjusted score
    score = ei / cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
