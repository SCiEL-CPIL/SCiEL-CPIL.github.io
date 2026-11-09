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
    Hybrid EI-LCB acquisition function.
    
    Args:
        mu: Mean of the GP posterior, shape (2000,)
        sigma: Standard deviation of the GP posterior, shape (2000,)
        f_best: Best observed function value so far.
        
    Returns:
        scores: Array of shape (2000,) where higher scores are preferred.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    # Handle edge cases for numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Calculate LCB component (exploration term)
    # For minimization, we want low mu and high sigma.
    # Score component: beta * sigma - mu
    beta = 2.0  # Exploration parameter
    lcb_score = beta * sigma_safe - mu
    
    # Hybrid Score: Weighted sum of EI and LCB score
    # EI is "expected improvement", LCB score is "potential value drop"
    # Both are "higher is better" for the acquisition.
    # We need to balance them.
    # If f_best is very good, EI might be small. LCB might drive exploration.
    # Let's use a dynamic weight or fixed weight.
    # Fixed weight 0.5 / 0.5 seemed promising in the hint.
    
    # To make them comparable, we might need to scale.
    # EI is roughly bounded by (f_best - min_possible).
    # LCB score is roughly bounded by (max_sigma * beta - min_mu).
    # In early stages, sigma is large, so LCB dominates.
    # In late stages, sigma is small, EI dominates.
    # This is a natural balance.
    
    alpha = 0.5
    scores = alpha * ei + (1 - alpha) * lcb_score
    
    # Deadline Guard
    if _check_deadline():
        # If we are running out of time, return a simple, fast metric
        # to ensure we don't crash or timeout.
        # Just return EI or random. EI is already computed, so it's free.
        return ei
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
