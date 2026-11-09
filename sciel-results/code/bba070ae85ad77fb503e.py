import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Upper Confidence Bound (UCB) acquisition function.
    
    Args:
        mu: Mean prediction from GP, shape (2000,).
        sigma: Standard deviation prediction from GP, shape (2000,).
        f_best: Best observed value so far.
        cost: Cost of evaluating each candidate, shape (2000,).
        
    Returns:
        Scores for each candidate, shape (2000,). Higher is better.
    """
    # Exploration constant. Tuned for Rastrigin with short length scale.
    # Higher kappa encourages exploration of high-uncertainty regions.
    kappa = 2.5
    
    # Ensure cost is strictly positive to avoid division by zero
    cost_safe = np.maximum(cost, 1e-6)
    
    # UCB score for minimization:
    # We want to minimize mu, so we use -mu.
    # We want to explore high sigma regions, so we add kappa * sigma.
    # We penalize high cost by dividing by cost.
    score = (-mu + kappa * sigma) / cost_safe
    
    return score
