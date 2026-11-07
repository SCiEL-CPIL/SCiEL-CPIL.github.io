"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Levy 2D.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions at candidate points
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions at candidate points
    f_best : float
        Best observed function value so far
    cost : ndarray, shape (2000,)
        Evaluation costs for each candidate point
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Track time to ensure we don't exceed deadline
    t_start = time.time()
    DEADLINE = 320.0  # Safety margin before 330s kill
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard normal CDF and PDF for Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware acquisition: EI divided by square root of evaluation cost
    # Using sqrt(cost) provides a softer penalty than 1/cost, allowing
    # exploration of higher-cost regions where the shifted minimum may reside
    scores = ei / np.sqrt(cost)
    
    # Ensure all scores are finite (safety check)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard: if we're getting close to the limit, return immediately
    if time.time() - t_start > (DEADLINE - time.time()) * 0.95:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
