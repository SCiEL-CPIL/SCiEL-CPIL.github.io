import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB).
    
    Parameters:
    mu: array of shape (2000,) - GP mean predictions
    sigma: array of shape (2000,) - GP standard deviation predictions
    f_best: float - best observed function value
    cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    # Ensure sigma is strictly positive to avoid numerical issues
    sigma = np.maximum(sigma, 1e-6)
    
    # Exploration parameter for UCB
    # A value of 1.5 encourages significant exploration, which is beneficial
    # for multimodal functions like Griewank to avoid local minima.
    beta = 1.5
    
    # UCB score for minimization: -mu - beta * sigma
    # This balances exploitation (low mu) and exploration (high sigma)
    ucb_scores = -mu - beta * sigma
    
    # Cost-awareness: divide by cost to prefer cheaper evaluations
    # Cost is >= 1.0, so this is safe
    scores = ucb_scores / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
