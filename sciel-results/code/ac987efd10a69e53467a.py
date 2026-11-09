"""Program for Bayesian optimization acquisition function."""
import time
import numpy as np
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
TIME_BUDGET = 310.0  # Leave 20s margin before 330s kill


def _time_remaining():
    return TIME_BUDGET - (time.time() - _START_TIME)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a parameterized model optimized for
    the Styblinski-Tang 1D minimization task.
    
    Args:
        mu: GP posterior mean, shape (N,)
        sigma: GP posterior std, shape (N,)
        f_best: Best observed value so far (lower is better)
        
    Returns:
        Scores for each candidate, shape (N,). Higher scores preferred.
    """
    # Safety checks
    sigma = np.asarray(sigma, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Clamp extreme sigma values
    sigma_safe = np.clip(sigma_safe, 1e-10, 1e6)
    
    # Pure GP-LCB acquisition for minimization:
    # score = -mu - kappa * sigma
    # Higher scores mean lower predicted objective with exploration bonus.
    # No normalization or hybrid weighting: the scale is intrinsic to the
    # objective units, so the lower bound is directly comparable across steps.
    kappa = 2.0
    scores = -mu - kappa * sigma_safe
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
