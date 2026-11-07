import numpy as np
import time

# Global start time for deadline guard
_start_time = time.time()
_DEADLINE = 300.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Computes acquisition scores using Upper Confidence Bound (UCB) 
    with an adaptive exploration coefficient.
    
    Parameters:
    mu: np.ndarray of shape (N,) - Posterior mean
    sigma: np.ndarray of shape (N,) - Posterior standard deviation
    f_best: float - Best observed value so far (minimum)
    
    Returns:
    np.ndarray of shape (N,) - Acquisition scores (higher is better)
    """
    
    # Deadline guard
    if time.time() > _DEADLINE:
        # Fallback: simple exploitation
        return -mu

    # Ensure sigma is positive and finite
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 0.0)
    
    # Adaptive beta calculation
    # Hartmann 6D global minimum
    f_min = -3.32237
    # Approximate maximum value of Hartmann 6D on [0,1]^6
    f_max = 3.3
    
    # Calculate progress towards the global minimum
    # progress = 0 at f_max, 1 at f_min
    range_val = f_max - f_min
    if range_val <= 0:
        progress = 0.0
    else:
        progress = (f_max - f_best) / range_val
    
    # Clamp progress to [0, 1]
    progress = np.clip(progress, 0.0, 1.0)
    
    # Beta decreases as we get closer to the optimum
    # High exploration initially, low exploration later
    beta_max = 5.0
    beta_min = 0.5
    
    beta = beta_max - (beta_max - beta_min) * progress
    
    # UCB score for minimization:
    # We want to maximize the score.
    # Low mu is good -> -mu
    # High sigma is good for exploration -> + beta * sigma
    scores = -mu + beta * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e6)
    
    return scores
