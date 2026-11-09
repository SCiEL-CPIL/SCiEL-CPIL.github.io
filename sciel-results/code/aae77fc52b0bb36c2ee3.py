"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # 300s limit, 10s safety margin

def acquisition(mu, sigma, f_best):
    """
    GP-UCB acquisition function for minimization with time-varying beta.
    
    Uses a structural GP-UCB formulation: score = -mu + beta_t * sigma
    where beta_t = sqrt(log(t+1)) approximated from f_best magnitude.
    
    Parameters:
    mu: array of shape (N,)
    sigma: array of shape (N,)
    f_best: scalar, lowest observed value
    
    Returns:
    scores: array of shape (N,), higher values indicate better candidates
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        return -mu + 2.0 * sigma
    
    # Infer number of evaluations from f_best magnitude
    # f_best starts around 0 and decreases; larger |f_best| suggests more iterations
    # Use a heuristic: t ~ max(1, int(10 * abs(f_best))) capped at 30
    t = max(1, int(min(30, 10 * abs(f_best))))
    
    # Time-varying beta: beta_t = sqrt(log(t + 1))
    # Scale up since we want meaningful exploration
    beta_t = 2.0 * np.sqrt(np.log(t + 1))
    
    # Compute GP-UCB score for minimization:
    # We want high score for low mu (exploitation) and high sigma (exploration)
    # score = -mu + beta_t * sigma
    sigma_safe = np.where(sigma > 1e-10, sigma, 0.0)
    score = -mu + beta_t * sigma_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
