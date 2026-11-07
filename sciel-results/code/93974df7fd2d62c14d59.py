import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Expected Improvement (EI) plus a small exploration bonus,
    normalized by the evaluation cost.
    """
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Exploration bonus: encourages visiting high-uncertainty regions
    # This helps counteract the cost penalty which might suppress exploration
    # in expensive but uncertain areas.
    exploration_bonus = 0.1 * sigma
    
    # Total score
    scores = (ei + exploration_bonus) / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
