"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Combines expected improvement (EI) with an exploration term (UCB-like),
    both normalized to comparable scales, and divided by cost.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Deadline guard: if we've been running too long, return a safe fallback
    # (this function is called repeatedly, so we track time across calls)
    if not hasattr(acquisition, '_start_time'):
        acquisition._start_time = time.time()
    
    elapsed = time.time() - acquisition._start_time
    if elapsed > 320:  # Safety margin before 330s limit
        # Return a simple fallback that prefers low cost and low mu
        return -mu / (cost + 1e-9)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # Clip EI to be non-negative
    ei = np.maximum(ei, 0.0)
    
    # UCB exploration term
    # Use a beta that balances exploration
    beta = 0.5
    ucb = mu + beta * sigma_safe
    
    # Normalize EI and UCB to comparable scales
    # Z-score both terms
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-9
    ei_norm = (ei - ei_mean) / ei_std
    
    ucb_mean = np.mean(ucb)
    ucb_std = np.std(ucb) + 1e-9
    ucb_norm = (ucb - ucb_mean) / ucb_std
    
    # Weighted combination: 75% exploitation (EI), 25% exploration (UCB)
    # Note: for minimization, lower UCB is better, so we negate it
    combined = 0.75 * ei_norm - 0.25 * ucb_norm
    
    # Divide by cost to get cost-aware score
    scores = combined / (cost + 1e-9)
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
