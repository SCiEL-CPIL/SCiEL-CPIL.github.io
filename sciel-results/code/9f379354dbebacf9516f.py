"""Program for Bayesian optimization acquisition function."""
import time
import numpy as np

# Global deadline guard
_START_TIME = time.time()
TIME_BUDGET = 325.0  # Leave 5s margin before 330s kill


def _time_remaining():
    return TIME_BUDGET - (time.time() - _START_TIME)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using pure Gaussian Process Lower Confidence Bound (GP-LCB).
    
    For minimization, the Lower Confidence Bound is defined as:
    LCB(x) = mu(x) - kappa * sigma(x)
    
    We want to find the x that minimizes LCB(x).
    Since the interface requires returning scores where HIGHER is better,
    we return the negative of the LCB:
    score(x) = -LCB(x) = -mu(x) + kappa * sigma(x)
    
    This formulation:
    1. Favors low mean (exploitation): -mu is large when mu is small.
    2. Favors high uncertainty (exploration): +kappa*sigma is large when sigma is large.
    
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
    
    # Ensure sigma is strictly positive and within reasonable bounds
    sigma_safe = np.maximum(sigma, 1e-10)
    sigma_safe = np.clip(sigma_safe, 1e-10, 1e6)
    
    # Fixed exploration parameter as per directive
    # kappa = 2.0 is a standard choice for LCB
    kappa = 2.0
    
    # Compute GP-LCB score for minimization (maximized form)
    # score = -mu + kappa * sigma
    scores = -mu + kappa * sigma_safe
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Time guard: if we're running out of time, return the scores immediately
    # The calculation is O(N) and very fast, so this is mostly a formality
    if _time_remaining() < 1.0:
        return scores
        
    return scores
