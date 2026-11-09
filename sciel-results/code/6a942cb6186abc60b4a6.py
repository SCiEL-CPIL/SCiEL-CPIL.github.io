import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB) with exploration-exploitation balance.
    """
    # Ensure sigma is not zero to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # UCB acquisition function
    # Beta controls the exploration-exploitation trade-off
    beta = 1.0
    ucb = -mu + beta * sigma
    
    # Add a small amount of noise to encourage diversity
    # This is a heuristic to avoid getting stuck in local optima
    noise = 0.01 * np.random.randn(len(ucb))
    
    return ucb + noise
