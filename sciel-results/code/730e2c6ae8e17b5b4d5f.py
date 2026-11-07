import numpy as np
import time

_start_time = time.time()
_DEADLINE_MARGIN = 15.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed function value
    cost : ndarray, shape (2000,)
        Cost of evaluating each point
    
    Returns:
    --------
    score : ndarray, shape (2000,)
        Acquisition scores (higher is better)
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 330.0 - _DEADLINE_MARGIN:
        return np.full(2000, 0.0)
    
    # Convert to numpy arrays and ensure finiteness
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Safe sigma for sampling
    eps = 1e-10
    safe_sigma = np.maximum(sigma, eps)
    
    # Thompson Sampling: Draw a sample from the posterior N(mu, sigma)
    z = np.random.randn(2000)
    y_sample = mu + safe_sigma * z
    
    # Cost-aware score: Maximize - (sampled_value / cost)
    # This encourages low function values and low costs
    score = -y_sample / cost
    
    # Final safety check
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
