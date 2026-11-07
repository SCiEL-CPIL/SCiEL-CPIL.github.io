"""New acquisition function using Cost-Adjusted Upper Confidence Bound."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB) normalized by cost.
    
    Score(x) = (beta * sigma(x) - mu(x)) / cost(x)
    
    This approach balances exploration (high sigma) and exploitation (low mu) 
    while penalizing high evaluation costs. It is distinct from Expected Improvement 
    by using a linear combination of mean and uncertainty rather than a probabilistic 
    improvement metric.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma and cost to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Exploration parameter
    # A higher beta favors exploration (uncertainty) over exploitation (mean).
    # In a multimodal 6D space with few initial points, aggressive exploration is beneficial.
    beta = 2.0
    
    # Calculate UCB score for minimization:
    # We want to minimize mu - beta * sigma.
    # To maximize the score (higher is better), we use:
    # score = -(mu - beta * sigma) / cost = (beta * sigma - mu) / cost
    scores = (beta * sigma - mu) / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
