# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a modified Expected Improvement (EI) with cost normalization,
    plus a small exploration bonus to avoid getting stuck.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean predictions from GP.
    sigma : array-like, shape (2000,)
        Standard deviation predictions from GP.
    f_best : float
        Best (lowest) observed function value.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate).
    """
    # Start with a deadline guard
    t_start = time.time()
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-score for expected improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-normalized EI
    scores = ei / cost_safe
    
    # Add a small exploration bonus based on uncertainty
    # This helps escape local optima
    exploration_bonus = 0.05 * sigma_safe / cost_safe
    
    scores = scores + exploration_bonus
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Check deadline
    elapsed = time.time() - t_start
    if elapsed > 280:  # Safety margin before 300s limit
        # Return the best result found so far
        return scores
    
    return scores
# EVOLVE-BLOCK-END
