"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track state across calls within a single seed
_state = {"call_count": 0, "seed_start_time": None}

def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function with adaptive beta.
    
    Uses a robust call counter that resets when f_best improves significantly
    or when we detect a new seed (via time gap or call count reset heuristic).
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        GP mean at candidate points.
    sigma : ndarray, shape (2000,)
        GP standard deviation at candidate points.
    f_best : float
        Best observed objective value so far.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    t_start = time.time()
    deadline = t_start + 250.0  # Safety margin: 330s total, leave 80s buffer
    
    # Clamp sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Simple, robust beta: use a fixed value that works well for BO
    # Based on literature, beta ~ 2-4 works well for early exploration
    # Let's use a moderate fixed beta
    beta = 2.5
    
    # UCB score for minimization: maximize (beta * sigma - mu)
    scores = beta * sigma_safe - mu
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
