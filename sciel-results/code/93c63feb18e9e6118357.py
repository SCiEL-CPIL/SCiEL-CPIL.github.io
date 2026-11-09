import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (PI).
    
    Parameters:
    mu: (2000,) array of posterior means.
    sigma: (2000,) array of posterior standard deviations.
    f_best: float, the best observed value so far.
    cost: (2000,) array of evaluation costs for each candidate.
    
    Returns:
    scores: (2000,) array of acquisition scores (higher is better).
    """
    # Avoid division by zero in sigma
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best)
    pi = norm.cdf(z)
    
    # Cost-aware score: PI divided by cost
    # Ensure cost is positive to avoid division issues, though it should be >= 1
    cost_safe = np.where(cost > 1e-9, cost, 1e-9)
    
    scores = pi / cost_safe
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
