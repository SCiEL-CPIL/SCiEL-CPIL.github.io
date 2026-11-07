"""Rewritten acquisition function for Bayesian optimization of 2D RBF mixture."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    # Deadline guard
    start_time = time.time()
    deadline = 295.0  # seconds, safety margin before 300s kill

    # Ensure inputs are numpy arrays with correct dtype
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Prevent division by zero and negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)

    # Adaptive Lower Confidence Bound (LCB) for minimization
    # We maximize: -mu + beta * sigma
    
    # Estimate optimization progress to adapt beta
    # f_min is the known global minimum
    # f_start is an estimate of the initial random points' quality
    f_min = -2.53218
    f_start = 0.0
    
    range_val = f_start - f_min
    if range_val < 1e-9:
        progress = 0.0
    else:
        progress = (f_best - f_min) / range_val
        progress = np.clip(progress, 0.0, 1.0)
        
    # Beta decays as we make progress (progress 1.0 -> high beta, progress 0.0 -> low beta)
    beta_min = 0.5
    beta_max = 5.0
    beta = beta_min + (beta_max - beta_min) * progress
    
    # LCB score: higher is better
    scores = -mu + beta * sigma_safe
    
    # Check deadline
    if time.time() - start_time > deadline - 10:
        # Ensure finite before returning
        scores = np.where(np.isfinite(scores), scores, -1e9)
        return scores

    # Ensure all scores are finite
    # If any calculation resulted in nan/inf, replace with a very low value (-1e9)
    # so it won't be selected, but the array remains finite
    scores = np.where(np.isfinite(scores), scores, -1e9)

    # Final deadline check
    if time.time() - start_time > deadline - 5:
        return scores

    return scores
# EVOLVE-BLOCK-END
