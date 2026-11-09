"""Acquisition function for Bayesian optimization of hartmann 6d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining LCB-style exploration with a soft exploitation term.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions at candidate points.
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions at candidate points.
    f_best : float
        Best (lowest) observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more promising to evaluate next.
    """
    # Guard against zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # LCB: mu - beta * sigma, where beta controls exploration
    # In high-dim multimodal problems with few samples, aggressive exploration helps
    # We use a time-adaptive beta that starts high and decays
    n_candidates = len(mu)
    
    # Estimate progress: if f_best is close to known minimum (-3.32237), we're doing well
    # Use a fixed beta that balances exploration/exploitation for 30 evals
    # LCB with beta ~ 1.5-2.0 works well for small sample sizes in 6D
    beta = 1.8
    
    # Primary score: LCB (lower mu + higher sigma is better)
    lcb_score = -mu + beta * sigma_safe
    
    # Add a small exploitation bonus for points that are already good
    # This helps avoid pure exploration when we're close to the optimum
    # The term (f_best - mu) / (|f_best| + 1) gives a normalized exploitation signal
    exploit_scale = abs(f_best) + 1.0
    exploit_bonus = 0.3 * (f_best - mu) / exploit_scale
    
    # Combine: LCB is primary, exploitation is secondary
    scores = lcb_score + exploit_bonus
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
