import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining expected improvement (EI)
    with an uncertainty bonus, normalized by evaluation cost.
    
    Parameters:
    -----------
    mu : ndarray, shape (n,)
        GP mean predictions.
    sigma : ndarray, shape (n,)
        GP standard deviation predictions.
    f_best : float
        Best observed objective value so far.
    cost : ndarray, shape (n,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores (higher = more desirable).
    """
    # Clip sigma to avoid division by zero or negative values
    sigma = np.maximum(sigma, 1e-12)
    cost = np.maximum(cost, 1e-12)
    
    # Standardized improvement
    z = (f_best - mu) / sigma
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Uncertainty bonus: encourage exploration where uncertainty is high
    # The bonus is scaled relative to the current best to keep it meaningful
    uncertainty_bonus = sigma * 0.5
    
    # Cost-aware score: balance expected improvement and exploration
    # The cost acts as a penalty for expensive evaluations
    scores = (ei + uncertainty_bonus) / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
