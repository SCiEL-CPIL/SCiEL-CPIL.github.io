import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB).
    
    Parameters:
    mu: array of shape (2000,), predicted mean
    sigma: array of shape (2000,), predicted standard deviation
    f_best: float, best value found so far
    
    Returns:
    scores: array of shape (2000,), higher is better
    """
    # Clip sigma to avoid numerical issues, though UCB is more stable than EI/PI
    sigma = np.maximum(sigma, 1e-9)
    
    # UCB coefficient. 
    # A value around 1.0 to 2.0 is typical for small budget BO.
    # We choose 1.5 to balance exploration and exploitation.
    beta = 1.5
    
    # We are minimizing f, so we want low mu and high sigma.
    # Score = -mu + beta * sigma
    scores = -mu + beta * sigma
    
    # Handle any non-finite values
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    
    return scores
