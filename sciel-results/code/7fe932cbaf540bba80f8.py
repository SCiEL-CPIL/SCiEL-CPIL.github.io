import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    mu: array of shape (2000,), mean of the GP surrogate.
    sigma: array of shape (2000,), standard deviation of the GP surrogate.
    f_best: float, the best observed value so far.
    cost: array of shape (2000,), cost of evaluating each candidate.
    
    Returns:
    scores: array of shape (2000,), scores for each candidate (higher is better).
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Avoid division by zero and numerical instability
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration bonus proportional to sigma
    # The inspiration suggests this improves balance.
    # We normalize by cost to respect the cost-awareness.
    beta = 0.5
    exploration = beta * sigma_safe
    
    # Combine EI and exploration, normalized by cost
    scores = (ei + exploration) / cost_safe
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
