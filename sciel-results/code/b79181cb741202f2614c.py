"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track evaluation count across calls using a global variable
_eval_count = [0]
_start_time = [None]


def acquisition(mu, sigma, f_best):
    """
    Adaptive acquisition function for Bayesian optimization of Styblinski-Tang 1D.
    
    Uses a hybrid approach:
    - Early: aggressive exploration (high kappa LCB)
    - Mid: balanced exploration
    - Late: exploitation-focused with a small exploration component
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Posterior mean predictions.
    sigma : ndarray, shape (2000,)
        Posterior standard deviation predictions.
    f_best : float
        Best observed value so far.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values are preferred.
    """
    global _eval_count, _start_time
    
    # Deadline guard
    if _start_time[0] is None:
        _start_time[0] = time.time()
    elapsed = time.time() - _start_time[0]
    # Kill if we're getting close to the 300s limit
    if elapsed > 290:
        # Return a simple exploitation-based score quickly
        scores = -mu
        scores = np.where(np.isfinite(scores), scores, -1e10)
        return scores
    
    _eval_count[0] += 1
    n_evals = _eval_count[0]
    
    # Total evaluations: 30 (5 initial + 25 acquisition steps)
    total_evals = 30
    progress = min(n_evals / total_evals, 1.0)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Adaptive kappa based on progress
    # Early: high exploration (kappa ~ 3.0)
    # Mid: moderate (kappa ~ 1.5)
    # Late: low exploration (kappa ~ 0.5)
    if progress < 0.25:
        kappa = 3.0
    elif progress < 0.5:
        kappa = 2.0
    elif progress < 0.75:
        kappa = 1.2
    else:
        kappa = 0.5
    
    # LCB score: maximize -mu + kappa * sigma
    scores = -mu + kappa * sigma_safe
    
    # Add a small diversity bonus to avoid getting stuck
    # Penalize points that are too close to where we've been (implicitly via GP)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
