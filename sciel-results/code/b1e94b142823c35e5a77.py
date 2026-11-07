"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 330.0  # wall-clock seconds total allowed


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Styblinski-Tang 1D.
    
    Uses Upper Confidence Bound (UCB) to balance exploitation and exploration.
    UCB is effective for multimodal functions as it explicitly rewards high
    uncertainty regions, helping to escape local optima.
    
    Args:
        mu: array of shape (2000,) with posterior means
        sigma: array of shape (2000,) with posterior standard deviations
        f_best: lowest observed function value
    
    Returns:
        array of shape (2000,) with acquisition scores (higher is better)
    """
    # Guard against negative or zero sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB parameter: controls exploration vs exploitation
    # Higher values encourage more exploration
    beta = 2.5
    
    # UCB formula for minimization: maximize -mu + beta * sigma
    score = -mu + beta * sigma_safe
    
    # Ensure finite output, replacing non-finite values with -inf
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
