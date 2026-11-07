import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Effective Probability of Improvement (CEPI).
    
    Args:
        mu: Mean of the GP posterior, shape (2000,)
        sigma: Standard deviation of the GP posterior, shape (2000,)
        f_best: Best observed function value
        cost: Cost of evaluating each candidate, shape (2000,)
        
    Returns:
        scores: Acquisition scores, shape (2000,)
    """
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for Probability of Improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best)
    pi = norm.cdf(z)
    
    # Cost-Effective Probability of Improvement
    # We divide by cost to prefer cheaper points that have a good chance of improvement
    # To avoid division by zero or very small costs (though cost >= 1), we use safe division
    cost_safe = np.maximum(cost, 1e-9)
    
    scores = pi / cost_safe
    
    # Ensure all values are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
