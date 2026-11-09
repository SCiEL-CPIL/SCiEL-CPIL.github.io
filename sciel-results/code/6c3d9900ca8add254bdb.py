import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Upper Confidence Bound (UCB) for minimization.
    
    Args:
        mu: Array of posterior means (shape 2000,).
        sigma: Array of posterior standard deviations (shape 2000,).
        f_best: The lowest observed value so far.
        
    Returns:
        Array of scores (shape 2000,). Higher score indicates higher priority.
    """
    start_time = time.time()
    # Safety margin: 300s limit, stop at 290s
    deadline = start_time + 290.0
    
    # UCB for minimization:
    # We want to select points with low expected value (mu) or high uncertainty (sigma).
    # Score = beta * sigma - mu
    # Higher score -> Lower mu or Higher sigma.
    
    # Beta = 1.0 is a standard constant for UCB in Bayesian Optimization contexts
    # where the number of steps is small (e.g., < 100).
    beta = 1.0
    
    # Compute scores
    scores = beta * sigma - mu
    
    # Check deadline
    if time.time() > deadline:
        # If we exceed the deadline, we return the current scores.
        # The computation is vectorized and very fast, so this is unlikely.
        pass
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
