import numpy as np
from scipy.stats import norm
import time

# Module-level state
_call_count = 0
_start_time = None

def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Pure GP-UCB with high fixed exploration coefficient.
    
    For minimization, we use: score = -mu + beta * sigma
    Higher scores indicate more promising points to evaluate next.
    
    Parameters:
    mu: np.ndarray, shape (N,)
        Predicted mean of the objective function.
    sigma: np.ndarray, shape (N,)
        Predicted standard deviation of the objective function.
    f_best: float
        The best (lowest) objective value observed so far.
        
    Returns:
    np.ndarray, shape (N,)
        Acquisition scores. Higher scores indicate more promising points.
    """
    global _call_count, _start_time
    _call_count += 1
    
    if _start_time is None:
        _start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to a small positive value to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Pure GP-UCB for minimization: score = -mu + beta * sigma
    # Use a high fixed beta for strong exploration
    # Based on inspiration 3: beta=5.0 works well
    beta = 5.0
    
    scores = -mu + beta * sigma_safe
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=-1e10, posinf=1e10, neginf=-1e10)
    
    return scores
