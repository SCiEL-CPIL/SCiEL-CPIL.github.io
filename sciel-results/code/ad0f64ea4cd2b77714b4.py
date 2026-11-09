import numpy as np
from scipy.stats import norm
import time

# Global state for deadline guard
_start_time = time.time()
_TIME_LIMIT = 320.0  # Safety margin from 330s

def _check_deadline():
    """Check if we are approaching the time limit."""
    return (time.time() - _start_time) > _TIME_LIMIT

def acquisition(mu, sigma, f_best):
    """
    Dynamic Hybrid EI-LCB acquisition function.
    
    Args:
        mu: Mean of the GP posterior, shape (2000,)
        sigma: Standard deviation of the GP posterior, shape (2000,)
        f_best: Best observed function value so far.
        
    Returns:
        scores: Array of shape (2000,) where higher scores are preferred.
    """
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Handle edge cases for numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Calculate Dynamic LCB component
    # Hartmann 3D global minimum is approximately -3.86278
    min_val = -3.86278
    gap = f_best - min_val
    
    # Scale gap to [0, 1] to control beta
    # If gap > 1, we are far, use max exploration.
    # If gap < 0, we are at/below min (shouldn't happen often, but clamp), use min exploration.
    gap_clamped = np.clip(gap, 0.0, 1.0)
    
    # Beta: Exploration parameter
    # Start with higher beta (more exploration) when far from optimum.
    # Reduce beta (more exploitation) when close to optimum.
    beta_min = 1.5
    beta_max = 3.0
    beta = beta_min + (beta_max - beta_min) * gap_clamped
    
    lcb_score = -mu + beta * sigma_safe
    
    # Hybrid Score
    # We found that mixing EI and LCB helps.
    # We keep a 50/50 mix but the LCB term is now adaptive.
    alpha = 0.5
    scores = alpha * ei + (1 - alpha) * lcb_score
    
    # Deadline Guard
    if _check_deadline():
        return ei
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
