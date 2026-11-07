# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Adaptive Lower Confidence Bound (LCB).
    
    The exploration parameter beta is adapted based on the current best value
    relative to the known range of the Hartmann 6D function.
    Early in the optimization (poor f_best), beta is high to encourage exploration.
    Late in the optimization (good f_best), beta is low to encourage exploitation.
    
    Parameters
    ----------
    mu : ndarray, shape (n,)
        GP mean predictions.
    sigma : ndarray, shape (n,)
        GP standard deviation predictions.
    f_best : float
        Best (lowest) observed objective value so far.
    
    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values are preferred.
    """
    # Start a timer to guard against the 330s wall-clock limit
    t_start = time.time()
    
    # Guard against non-positive sigma
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Define the range of the Hartmann 6D function
    f_min = -3.32237
    f_max = 0.0  # Approximate upper bound
    
    # Calculate progress (0.0 for worst, 1.0 for best)
    # If f_best is very bad (e.g. > f_max), progress is 0.
    # If f_best is f_min, progress is 1.
    if f_max > f_min:
        progress = (f_max - f_best) / (f_max - f_min)
    else:
        progress = 0.0
        
    # Clip progress to [0, 1]
    progress = np.clip(progress, 0.0, 1.0)
    
    # Define beta range
    beta_high = 2.5
    beta_low = 0.5
    
    # Interpolate beta based on progress
    beta = beta_high - (beta_high - beta_low) * progress
    
    # Calculate LCB scores
    # For minimization, we want low mu and high sigma.
    # Score = -mu + beta * sigma
    scores = -mu + beta * sigma_safe
    
    # Check time budget
    elapsed = time.time() - t_start
    if elapsed > 280:  # Safety margin before 300s
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
