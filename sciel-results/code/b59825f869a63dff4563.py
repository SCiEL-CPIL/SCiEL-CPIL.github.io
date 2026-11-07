import numpy as np
import time

# Global state for deadline guard
_start_time = time.time()
_TIME_LIMIT = 300.0  # Safety margin before the 330s hard limit

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of hartmann 6d.
    Uses Lower Confidence Bound (LCB) with a fixed kappa for exploration.
    """
    global _start_time
    
    # Deadline guard: If we are running out of time, return a safe, valid score.
    # Returning zeros is valid (finite) and will select the first candidate.
    if time.time() - _start_time > _TIME_LIMIT:
        return np.zeros_like(mu, dtype=np.float64)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Structural Change: Lower Confidence Bound (LCB) for minimization.
    # We want to maximize the score.
    # LCB estimates the lower bound of the function value: mu - kappa * sigma.
    # We want to find points with LOW function values.
    # So we want to MAXIMIZE the "potential to be low".
    # Score = - (mu - kappa * sigma) = -mu + kappa * sigma.
    
    # Kappa controls the trade-off between exploitation (mu) and exploration (sigma).
    # A higher kappa encourages more exploration.
    # For 6D Hartmann, a moderate-to-high kappa is appropriate to escape local minima.
    kappa = 2.0
    
    scores = -mu + kappa * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
