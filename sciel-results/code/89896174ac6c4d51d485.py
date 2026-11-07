"""Acquisition function for Bayesian optimization of styblinski tang 1d.
Uses GP-UCB (Gaussian Process Upper Confidence Bound) with a dynamic beta
parameter that decays as the best observed value improves, balancing
exploration and exploitation for the 30-evaluation budget.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level deadline guard
_start_time = time.time()
_DEADLINE = _start_time + 300.0  # 300s wall-clock limit per evaluation


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP posterior mean at candidates (in objective units)
    sigma : array of shape (2000,)
        GP posterior standard deviation at candidates (in objective units)
    f_best : float
        Best (lowest) observed value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores; higher means evaluate next
    """
    # Deadline safety check
    if time.time() > _DEADLINE - 5.0:
        # Fallback: Simple UCB with fixed beta
        sigma_safe = np.maximum(sigma, 1e-9)
        return -mu + np.sqrt(10.0) * sigma_safe

    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Dynamic beta for UCB
    # f_best starts around 0-10 and goes to -39.16
    # We want high beta (exploration) when f_best is poor (high value)
    # and low beta (exploitation) when f_best is good (low value)
    
    # Map f_best from [10, -40] to progress [0, 1]
    # progress 0 -> bad, progress 1 -> good
    # Let's use a reference range.
    f_min_ref = -39.16
    f_max_ref = 10.0
    
    # Normalize f_best to [0, 1] where 1 is best
    # If f_best is -39, t=1. If f_best is 10, t=0.
    t = (f_max_ref - f_best) / (f_max_ref - f_min_ref)
    t = np.clip(t, 0.0, 1.0)
    
    # Beta schedule:
    # Start with high exploration (beta=20) and end with lower exploration (beta=5)
    # This encourages finding the global minimum early and refining it later.
    beta_max = 20.0
    beta_min = 5.0
    beta = beta_min + (beta_max - beta_min) * (1.0 - t)
    
    # Calculate UCB score for minimization:
    # We want to maximize -mu + sqrt(beta) * sigma
    # High -mu means low mu (good prediction)
    # High sigma means high uncertainty (good exploration)
    scores = -mu + np.sqrt(beta) * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
