"""Cost-aware acquisition function for Bayesian optimization (Styblinski-Tang 2D)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global time tracking for deadline guard
_start_time = time.time()
# The evaluator kills at 330s. We should finish acquisition calls well before.
# However, acquisition is called in a loop. We don't know the loop frequency.
# We assume the heavy work (GP) is done by the harness.
# We set a conservative limit for the *function call* itself.
# If the function takes > 10s, something is wrong. But 2000 points is fast.
# The deadline guard is more about ensuring we don't crash if the system is slow.
_CALL_TIMEOUT = 5.0  # Seconds


def _is_time_exceeded():
    return (time.time() - _start_time) > 300  # Hard limit is 330, leave margin


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Uses Cost-Adjusted Expected Improvement (EI / cost).
    This metric directly optimizes the expected improvement per unit of cost,
    which is optimal for a fixed total cost budget.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher is better)
    """
    # Deadline Guard: If we are close to the global timeout, return a simple score
    # to ensure we don't hang the evaluator. This is a safety net.
    if _is_time_exceeded():
        # Return a score that prefers cheap points, as a last resort
        # This is unlikely to be hit if the harness is efficient.
        return 1.0 / np.maximum(cost, 1e-9)

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safe sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Safe cost to avoid division by zero
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-Adjusted EI: EI / cost
    # This prioritizes points that offer high improvement for low cost.
    score = ei / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Ensure non-negative
    score = np.maximum(score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
