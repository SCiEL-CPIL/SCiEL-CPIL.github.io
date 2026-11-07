"""Thompson Sampling acquisition for Bayesian optimization of hartmann 3d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_START_TIME = None


def _ensure_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Draws a single sample from the joint Gaussian posterior over the 2000 candidates
    and returns the negative of this sample as the score. Higher score means
    the point is predicted to have a lower (better) objective value.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Posterior mean at each candidate point.
    sigma : array of shape (2000,)
        Posterior standard deviation at each candidate point.
    f_best : float
        Best observed objective value so far.
    
    Returns:
    --------
    scores : array of shape (2000,)
        Thompson sampling scores; higher means evaluate next.
    """
    _ensure_start_time()
    
    n = len(mu)
    
    # Clamp sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Draw a single sample from the joint Gaussian posterior:
    # f_sample = mu + sigma * z, where z ~ N(0, 1)
    z = np.random.standard_normal(n)
    f_sample = mu + sigma_safe * z
    
    # Return negative sample as score (higher score = lower predicted value)
    scores = -f_sample
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
