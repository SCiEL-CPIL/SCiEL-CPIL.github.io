"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 10s safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Linear Lower Confidence Bound (LBC)
    with cost normalization.
    
    score = (f_best - mu + beta * sigma) / (cost + eps)
    
    Parameters:
    -----------
    mu : array-like, shape (n,)
        GP mean predictions
    sigma : array-like, shape (n,)
        GP standard deviation predictions
    f_best : float
        Best (lowest) observed value so far
    cost : array-like, shape (n,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition values (higher = more desirable to evaluate)
    """
    # Deadline guard: return simple fallback to avoid timeout
    if time.time() > _DEADLINE:
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-9)
        score = (f_best - mu + 1.5 * sigma_safe) / (np.asarray(cost, dtype=np.float64) + 1e-9)
        return np.where(np.isfinite(score), score, 0.0)

    # Convert to numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 1e-9)

    # LBC score: exploitation + exploration, normalized by cost
    # beta=1.5 balances exploration well for this task
    beta = 1.5
    score = (f_best - mu + beta * sigma_safe) / (cost + 1e-9)

    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
