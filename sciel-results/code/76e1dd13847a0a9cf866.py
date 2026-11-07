import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using UCB-like optimistic improvement.
    
    Args:
    mu: array of mean predictions (shape 2000,)
    sigma: array of standard deviation predictions (shape 2000,)
    f_best: scalar, best observed value
    cost: array of evaluation costs (shape 2000,)
    
    Returns:
    scores: array of acquisition scores (shape 2000,), higher is better
    """
    # UCB parameter: trade-off between exploitation (mu) and exploration (sigma)
    # For minimization, we want to maximize (f_best - mu + beta * sigma)
    beta = 2.0
    
    # Ensure sigma is positive to avoid division issues or negative exploration terms
    # although sigma is a std dev, it should be >= 0
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Calculate optimistic improvement score
    # f_best - mu is the expected improvement if sigma=0
    # beta * sigma adds an exploration bonus
    scores = (f_best - mu + beta * sigma) / cost
    
    # Sanitize output to ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
