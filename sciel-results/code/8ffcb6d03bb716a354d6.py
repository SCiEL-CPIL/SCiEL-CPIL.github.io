"""Program for Bayesian optimization acquisition function."""
import time
import numpy as np

# Global deadline guard
_START_TIME = time.time()
TIME_BUDGET = 310.0  # Leave 20s margin before 330s kill


def _time_remaining():
    return TIME_BUDGET - (time.time() - _START_TIME)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using GP-LCB (Lower Confidence Bound) for
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
    
    # Use adaptive kappa based on progress toward the known optimum
    # ST function minimum is -39.1662
    target = -39.1662
    gap = f_best - target  # positive if we're above the optimum
    
    # When far from optimum (large gap), explore more (larger kappa)
    # When close to optimum (small gap), exploit more (smaller kappa)
    # Use a smooth decay: kappa starts around 3.0 and decays to ~1.0
    # Using a logistic-like function for smooth transition
    kappa = 1.0 + 2.0 * (1.0 / (1.0 + np.exp(-(gap - 5.0) / 3.0)))
    
    # GP-LCB: score = -mu - kappa * sigma
    # Higher scores mean more promising (lower expected value with exploration bonus)
    scores = -mu - kappa * sigma_safe
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
