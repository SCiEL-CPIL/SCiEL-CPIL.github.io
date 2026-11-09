"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track time to ensure we stay within budget
_start_time = time.time()
_TIME_LIMIT = 325.0  # 5 seconds margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Gaussian Process Lower Confidence Bound (GP-LCB) acquisition function.
    For minimization, we want to find points where the lower bound of the
    objective is lowest. The lower confidence bound is:
        LCB(x) = mu(x) - kappa * sigma(x)
    We want to MINIMIZE LCB, so the acquisition score (to maximize) is:
        score = -mu(x) + kappa * sigma(x)
    
    Parameters:
    mu : np.ndarray
        Posterior mean at candidate points, shape (N,).
    sigma : np.ndarray
        Posterior standard deviation at candidate points, shape (N,).
    f_best : float
        Best observed value so far (minimum).
    
    Returns:
    scores : np.ndarray
        LCB-based scores, shape (N,), higher is better.
    """
    global _start_time
    
    # Safety check: if we're close to the time limit, return a simple score
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        # Return deterministic score based on -mu to avoid hanging
        return -np.asarray(mu)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle numerical stability for sigma
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # GP-LCB for minimization:
    # Lower bound of f(x) is mu(x) - kappa * sigma(x)
    # We want to minimize this lower bound, so we maximize:
    #   score = -(mu - kappa * sigma) = -mu + kappa * sigma
    
    # Use kappa = 2.0 as a fixed exploration parameter
    # This balances exploration (high sigma) and exploitation (low mu)
    kappa = 2.0
    
    scores = -mu + kappa * sigma_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
