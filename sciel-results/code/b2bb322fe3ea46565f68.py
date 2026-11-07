import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Cost-aware UCB for minimization
    beta = 3.0
    
    # Clip sigma to avoid division by zero or numerical issues
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate UCB score
    # We want to minimize, so lower mu is better (hence -mu)
    # Higher sigma is better for exploration
    ucb = -mu + beta * sigma
    
    # Divide by cost
    scores = ucb / cost
    
    # Handle non-finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
