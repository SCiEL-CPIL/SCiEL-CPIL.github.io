import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI)
    divided by cost.
    
    Args:
        mu: (2000,) array of GP mean predictions.
        sigma: (2000,) array of GP standard deviations.
        f_best: float, the best observed value so far.
        cost: (2000,) array of costs for evaluating each candidate.
        
    Returns:
        scores: (2000,) array of acquisition scores (higher is better).
    """
    # Avoid division by zero in alpha calculation
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Standardized improvement alpha
    # For minimization, improvement is f_best - mu
    alpha = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # E[max(0, f_best - f)] where f ~ N(mu, sigma^2)
    # = (f_best - mu) * Phi(alpha) + sigma * phi(alpha)
    ei = (f_best - mu) * norm.cdf(alpha) + sigma_safe * norm.pdf(alpha)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: Efficiency of improvement
    # Divide by cost to prefer cheaper points with similar improvement potential
    # cost is in [1, 2] based on the problem description
    cost_safe = np.maximum(cost, 1e-8)
    
    scores = ei / cost_safe
    
    # Handle any numerical issues
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    return scores
