import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (POI).
    
    Args:
        mu: Array of shape (2000,) containing GP mean predictions.
        sigma: Array of shape (2000,) containing GP standard deviations.
        f_best: Scalar, the best function value observed so far.
        cost: Array of shape (2000,) containing the evaluation cost for each candidate.
        
    Returns:
        Array of shape (2000,) with acquisition scores. Higher is better.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is non-negative and add small epsilon for numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Z-score
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best)
    # Using the CDF of the standard normal distribution
    poi = norm.cdf(z)
    
    # Cost-aware score: POI / cost
    # We divide by cost to prefer cheaper evaluations with similar probability of improvement
    # Add small epsilon to cost to avoid division by zero (though cost >= 1)
    cost_safe = cost + 1e-9
    
    scores = poi / cost_safe
    
    # Ensure all scores are finite
    # Replace any non-finite values (NaN, Inf) with 0.0
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
