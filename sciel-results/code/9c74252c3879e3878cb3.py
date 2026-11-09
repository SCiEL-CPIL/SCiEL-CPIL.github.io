import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Scale factor to balance function values and cost
    scale = 10.0
    beta = 2.5
    lambda_cost = 1.0
    
    # Normalized cost: range [0, 1]
    norm_cost = cost - 1.0
    
    # Cost-aware UCB
    # Exploitation: f_best - mu
    # Exploration: beta * sigma
    # Penalty: lambda * norm_cost
    
    scores = (f_best - mu + beta * sigma) / scale - lambda_cost * norm_cost
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
