import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB) for minimization.
    
    Parameters:
    mu: Array of mean predictions (shape 2000,)
    sigma: Array of standard deviations (shape 2000,)
    f_best: Best observed value (scalar)
    
    Returns:
    scores: Array of scores (shape 2000,), higher is better.
    """
    # UCB for minimization: Score = -mu + beta * sigma
    # beta controls exploration vs exploitation
    beta = 3.0
    
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    # Replace any non-finite values with a very low number
    scores = np.where(np.isfinite(scores), scores, -1e15)
    
    return scores
