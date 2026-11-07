import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function for minimization.
    Uses a fixed beta of 4.0 to encourage exploration.
    
    Parameters:
    mu: array of shape (2000,) with posterior mean.
    sigma: array of shape (2000,) with posterior standard deviation.
    f_best: scalar, the best value observed so far.
    
    Returns:
    array of shape (2000,) with scores. Higher score means more likely to be selected.
    """
    start_time = time.time()
    deadline = start_time + 290.0 # Safety margin before 300s limit

    # Ensure sigma is positive to avoid division by zero (though not strictly needed for UCB)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB for minimization:
    # We want to find points with low values.
    # The lower confidence bound is mu - beta * sigma.
    # We want to maximize the "optimism" or rather, we want to select the point 
    # that has the lowest potential value.
    # So we score by: - (mu - beta * sigma) = -mu + beta * sigma
    # Higher score = lower (mu - beta*sigma) = better candidate.
    
    beta = 4.0
    scores = -mu + beta * sigma_safe
    
    # Check time (UCB is O(N), so it's very fast)
    # No complex logic needed, just return
    
    return scores
