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
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Clamp extreme sigma values
    sigma_safe = np.clip(sigma_safe, 1e-10, 1e6)
    
    # Use a fixed exploration parameter
    # kappa controls the trade-off between exploitation and exploration
    # Higher kappa = more exploration, lower kappa = more exploitation
    # Start with a moderate value that balances both
    kappa = 2.0
    
    # Compute LCB score: -mu + kappa * sigma
    # Higher scores are preferred (we want to maximize this)
    scores = -mu + kappa * sigma_safe
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Time guard: if we're running out of time, return the scores as-is
    # (they are already computed and simple)
    return scores
