"""Acquisition function for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Track time to ensure we don't exceed the 300s limit (with safety margin)
_start_time = None

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed value so far.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores, higher meaning more promising.
    """
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: return early if we're close to the time limit
    elapsed = time.time() - _start_time
    if elapsed > 290:  # 10s safety margin before 300s limit
        # Return a simple score based on mean to be safe
        return -mu
    
    # Handle numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Use a combination of Expected Improvement and UCB-like exploration
    # This balances exploitation (EI) with exploration (uncertainty)
    
    # Standard EI
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Normalize EI to [0, 1] roughly for blending
    ei_max = np.max(ei)
    if ei_max > 0:
        ei_norm = ei / ei_max
    else:
        ei_norm = ei
    
    # UCB-like term: lower mean + beta * sigma
    # Use a moderate beta that decreases over time isn't possible without step count,
    # so use a fixed exploration weight
    beta = 0.5
    ucb = -mu + beta * sigma_safe
    ucb = np.where(np.isfinite(ucb), ucb, 0.0)
    
    # Normalize UCB
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = ucb_max - ucb_min
    if ucb_range > 0:
        ucb_norm = (ucb - ucb_min) / ucb_range
    else:
        ucb_norm = ucb - ucb_min
    
    # Blend: primarily EI, with a small UCB boost for exploration
    # Weight EI more heavily as it's the standard for minimization
    weights_ei = 0.7
    weights_ucb = 0.3
    
    scores = weights_ei * ei_norm + weights_ucb * ucb_norm
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
