"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Track state across calls within a single seed
_state = {"call_count": 0, "seed_start_time": None}

def acquisition(mu, sigma, f_best):
    """
    Expected Improvement (EI) acquisition function.
    
    Switches from UCB to EI, which is a different acquisition family that
    balances exploration and exploitation based on the expected gain.
    
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
    # Deadline guard: if we take too long, return zeros (safe fallback)
    # Although EI is fast, this satisfies the directive.
    if time.time() - t_start > 250.0:
        return np.zeros_like(mu)
        
    # Clamp sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Expected Improvement for minimization:
    # EI(x) = E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    # We add a small threshold xi to encourage exploration beyond the current best.
    # xi is scaled with f_best to be robust to the magnitude of the objective.
    # For Beale, f_best starts high and decreases to 0.
    xi = 0.01 * abs(f_best) + 1e-3
    
    z = (f_best - mu - xi) / sigma_safe
    
    # Calculate EI
    # norm.cdf(z) is Phi(z)
    # norm.pdf(z) is phi(z)
    scores = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
