import numpy as np

def acquisition(mu, sigma, f_best, cost):
    # Ensure inputs are numpy arrays for vectorized operations
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Small epsilon for numerical stability
    eps = 1e-9
    
    # UCB acquisition: mu + beta * sigma
    # beta=2.0 was a good baseline. Let's try 2.5 for a bit more exploration
    # as Shekel 4D is multimodal.
    beta = 2.5
    
    ucb = mu + beta * sigma
    
    # Cost-aware score: UCB / cost
    # Cost is provided in the 'cost' array (shape 2000)
    # We want to maximize this score.
    # Since cost >= 1, dividing by cost penalizes expensive candidates.
    
    scores = ucb / (cost + eps)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
