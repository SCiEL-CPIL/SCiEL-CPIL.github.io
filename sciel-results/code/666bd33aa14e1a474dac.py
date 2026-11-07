import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB).
    
    Parameters:
    mu (np.ndarray): Mean of the GP posterior, shape (N,).
    sigma (np.ndarray): Standard deviation of the GP posterior, shape (N,).
    f_best (float): The best observed value so far (minimum).
    
    Returns:
    np.ndarray: Acquisition scores, shape (N,). Higher is better.
    """
    # UCB score for minimization: -mu + beta * sigma
    # We use beta = 2.0 to encourage exploration.
    beta = 2.0
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    
    scores = -mu + beta * sigma
    
    # Ensure finite output
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e9, neginf=-1e9)
    
    return scores
