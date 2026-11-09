"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses Probability of Improvement (PI) divided by cost, with a small
    exploration bonus to avoid degenerate behavior when sigma is very small.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean predictions from the GP
    sigma : ndarray, shape (2000,)
        Standard deviation predictions from the GP
    f_best : float
        Best observed value so far
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Track elapsed time for safety margin
    t_start = time.time()
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Probability of Improvement: P(f(x) < f_best)
    # = P(Z < (f_best - mu) / sigma) where Z ~ N(0,1)
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    
    # Add a small exploration bonus proportional to sigma to encourage
    # exploring uncertain regions, but keep it small so PI dominates
    exploration_bonus = 0.05 * sigma_safe
    
    # Combine PI with exploration bonus
    combined = pi + exploration_bonus
    
    # Divide by cost to get cost-aware score
    scores = combined / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Safety check: if we're taking too long, return early
    elapsed = time.time() - t_start
    if elapsed > 280:  # 20 second safety margin
        return scores
    
    return scores
# EVOLVE-BLOCK-END
