import numpy as np
import time

_START_TIME = time.time()
_TIMEOUT = 325.0  # Leave a safety margin before the 330s limit

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    
    Args:
        mu: Array of shape (2000,) with posterior means.
        sigma: Array of shape (2000,) with posterior standard deviations.
        f_best: The lowest observed function value so far.
        
    Returns:
        Array of shape (2000,) with scores. Higher scores indicate better candidates.
    """
    # Deadline guard: If we are running out of time, return a simple, fast score
    # to ensure we don't timeout. A simple LCB with k=1 is fast.
    if time.time() - _START_TIME > _TIMEOUT:
        sigma_safe = np.maximum(sigma, 1e-8)
        return -mu + 1.0 * sigma_safe

    # Structural Change: Pure Lower Confidence Bound (LCB) strategy.
    # For minimization, we want to maximize the potential for finding a lower value.
    # The LCB is mu - k * sigma. We maximize this, so we return -(mu - k * sigma) = -mu + k * sigma.
    
    # Use a fixed, high exploration constant as per directive and knowledge base findings.
    # k=4.0 was found to improve performance in a related context.
    k = 4.0
    
    # Ensure sigma is strictly positive to avoid numerical issues, though GP sigma should be > 0.
    # 1e-8 is a safe lower bound.
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate the LCB score
    score = -mu + k * sigma_safe
    
    # Sanitize output to ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    score = np.clip(score, -1e6, 1e6)
    
    return score
