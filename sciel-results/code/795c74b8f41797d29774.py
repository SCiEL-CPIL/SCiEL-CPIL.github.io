import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB)
    for minimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP at each candidate point.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP at each candidate point.
    f_best : float
        Best observed value so far.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more promising).
    """
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-6)
    
    # UCB parameter for exploration
    # A common heuristic is beta = 2 or beta = sqrt(2 * log(t))
    # Since t is not directly available, we use a fixed value.
    # Tuning this might be necessary, but 2.0 is a standard default.
    beta = 2.0
    
    # For minimization, we maximize the lower confidence bound:
    # LCB(x) = mu(x) - beta * sigma(x)
    # To maximize the *potential* for a low value, we maximize -LCB(x)
    # or equivalently maximize beta * sigma(x) - mu(x)
    ucb_score = beta * sigma - mu
    
    # Cost-aware adjustment
    # We divide by cost to prefer cheaper evaluations.
    # Add epsilon to avoid division by zero.
    cost_safe = np.maximum(cost, 1e-6)
    
    scores = ucb_score / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
