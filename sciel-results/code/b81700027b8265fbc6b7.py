"""Acquisition function for Bayesian optimization of branin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state for deadline tracking
_start_time = time.time()
_TIME_LIMIT = 330  # seconds


def acquisition(mu, sigma, f_best):
    """
    Adaptive Expected Improvement (EI) acquisition function for minimization.
    Uses a time-based deadline guard and an adaptive xi parameter to balance
    exploration and exploitation based on the current best observed value.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions at candidate points.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions at candidate points.
    f_best : float
        Best observed value so far (lowest).
    
    Returns
    -------
    scores : np.ndarray, shape (2000,)
        Finite scores; higher means evaluate next.
    """
    # Deadline guard: if we are close to the time limit, return a safe,
    # exploitation-focused score to ensure the loop can finish cleanly.
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT - 10:
        # Fallback: simple UCB with small exploration to avoid division by zero issues
        sigma_safe = np.maximum(sigma, 1e-9)
        scores = -mu + 0.01 * sigma_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    # Known minimum of Branin 2D
    f_min = 0.397887
    
    # Adaptive xi parameter:
    # When f_best is far from the optimum, use a larger xi to encourage
    # significant improvements (exploration of new basins).
    # When f_best is close to the optimum, use a smaller xi for fine-grained exploitation.
    # Heuristic thresholds based on typical Branin function values.
    if f_best > 50.0:
        xi = 1.0
    elif f_best > 10.0:
        xi = 0.5
    elif f_best > 2.0:
        xi = 0.1
    else:
        xi = 0.0

    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Z-score for improvement with offset xi
    z = (f_best - xi - mu) / sigma_safe
    
    # Expected Improvement for minimization with offset:
    # EI_xi = (f_best - xi - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - xi - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clip negative EI values to 0
    ei = np.maximum(ei, 0.0)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(ei), ei, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
