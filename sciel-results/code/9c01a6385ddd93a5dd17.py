import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Upper Confidence Bound (UCB) acquisition function for minimization.
    
    Args:
        mu: Array of shape (2000,) with GP mean predictions.
        sigma: Array of shape (2000,) with GP standard deviations.
        f_best: Scalar, the best observed value so far.
        cost: Array of shape (2000,) with evaluation costs.
        
    Returns:
        Array of shape (2000,) with scores (higher is better).
    """
    # Ensure sigma is non-negative and safe for operations
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB for minimization:
    # We want to minimize f(x).
    # The lower confidence bound is L(x) = mu(x) - beta * sigma(x).
    # We want to select x that minimizes L(x).
    # This is equivalent to maximizing -L(x) = -mu(x) + beta * sigma(x).
    
    # Beta controls the exploration/exploitation trade-off.
    # Higher beta -> more exploration.
    beta = 2.0
    
    # Calculate UCB score
    score = -mu + beta * sigma_safe
    
    # Adjust for cost: divide by cost to prefer cheaper points
    # Cost is in [1, 2], so this is a moderate penalty
    score = score / cost
    
    # Ensure the output is finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
