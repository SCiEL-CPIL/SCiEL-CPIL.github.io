"""Acquisition function for Bayesian optimization of Beale 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global timer for deadline guard
_start_time = time.time()
_DEADLINE = 300.0  # Safety margin from 330s kill

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Instead of using a deterministic score like EI or UCB, we draw a sample
    from the posterior distribution of the objective function at the candidate
    points and select the one with the lowest sampled value. This naturally
    balances exploration and exploitation and is a structurally different
    approach from the previous EI/UCB hybrid.
    
    mu: array of shape (N,) - posterior mean
    sigma: array of shape (N,) - posterior std
    f_best: scalar - best observed value so far
    
    Returns: array of shape (N,) with higher scores preferred.
             The score is -sampled_value, so the evaluator picks the max score (min sampled value).
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    
    # Safety: clip sigma to avoid division by zero or extreme values
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Check deadline
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - 5.0:
        # If we are close to the deadline, return a simple deterministic fallback
        # to ensure we don't timeout. We prefer low mu.
        return -mu
    
    # Generate random samples from the posterior: y = mu + sigma * z
    # z ~ N(0, 1)
    z = np.random.standard_normal(len(mu))
    
    # Sampled values
    y_samples = mu + sigma_safe * z
    
    # We want to minimize y_samples, so the score should be -y_samples
    # Higher score means lower sampled value, which is preferred.
    score = -y_samples
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
