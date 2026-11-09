"""Acquisition function for Bayesian optimization of ackley 2d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE_MARGIN = 10.0
_TOTAL_BUDGET = 330.0


def _time_remaining():
    return _TOTAL_BUDGET - _DEADLINE_MARGIN - (time.time() - _start_time)


def _expected_improvement_min(mu, sigma, f_best):
    """
    Expected Improvement for minimization.
    EI = E[max(f_best - f, 0)] where f ~ N(mu, sigma^2)
    """
    sigma_safe = np.maximum(sigma, 1e-12)
    improvement = f_best - mu
    z = improvement / sigma_safe
    # For minimization: EI = improvement * Phi(z) + sigma * phi(z)
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return np.maximum(ei, 0.0)


def acquisition(mu, sigma, f_best):
    """
    Expected Improvement acquisition function for minimizing Ackley 2D.
    
    Uses the standard EI criterion with a small exploration bonus proportional
    to uncertainty, balancing exploitation and exploration.
    
    Parameters:
        mu: array of shape (2000,) - posterior mean
        sigma: array of shape (2000,) - posterior std
        f_best: float - best observed value so far
    
    Returns:
        scores: array of shape (2000,) - higher means evaluate next
    """
    # Deadline guard: if time is running out, return pure exploitation
    if _time_remaining() < 5.0:
        return -mu.copy()
    
    n = len(mu)
    
    # Compute Expected Improvement for minimization
    ei = _expected_improvement_min(mu, sigma, f_best)
    
    # Add a small exploration bonus: reward high uncertainty
    # Scale by the typical scale of EI to keep it proportional
    ei_scale = np.mean(ei) if np.mean(ei) > 1e-12 else 1.0
    exploration_bonus = 0.1 * (sigma / ei_scale)
    
    # Combine: higher EI is better, plus small exploration bonus
    scores = ei + exploration_bonus
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
