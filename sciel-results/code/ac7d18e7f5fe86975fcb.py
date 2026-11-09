import numpy as np
import time

_start_time = time.time()
TIME_LIMIT = 300.0 # Safety margin from 330s
_call_count = 0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Time-Adaptive Upper Confidence Bound (UCB).
    
    The UCB score is given by:
    score = -mu + beta * sigma
    
    beta is adapted based on the number of acquisition calls, which correlates 
    with the number of BO iterations. Early on, beta is high to encourage 
    exploration. Later, beta is lower to focus on exploitation.
    
    Parameters:
    mu (np.array): Mean of the GP posterior, shape (2000,)
    sigma (np.array): Standard deviation of the GP posterior, shape (2000,)
    f_best (float): Best observed function value so far.
    
    Returns:
    np.array: Scores for the 2000 candidates, shape (2000,). Higher is better.
    """
    global _call_count
    _call_count += 1
    
    # Deadline guard
    if time.time() - _start_time > TIME_LIMIT:
        # Fallback to a fast, deterministic method if time is critical
        # Return negative mu (exploitation)
        return -mu

    # Total BO steps are 25 (30 total evals - 5 initial).
    # We assume each call to acquisition corresponds to one step.
    # Clamp the fraction to [0, 1]
    total_steps = 25
    step_frac = min(1.0, max(0.0, _call_count / total_steps))
    
    # Time-adaptive beta:
    # Start with beta=4.0 (exploration) and end with beta=1.0 (exploitation)
    # Linear interpolation
    beta_start = 4.0
    beta_end = 1.0
    beta = beta_start + (beta_end - beta_start) * step_frac
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # Calculate UCB score
    # We want to minimize the function, so we want low mu.
    # We want to explore, so we want high sigma.
    # Score = -mu + beta * sigma
    scores = -mu + beta * sigma
    
    # Ensure scores are finite
    scores[~np.isfinite(scores)] = 0.0
    
    return scores
