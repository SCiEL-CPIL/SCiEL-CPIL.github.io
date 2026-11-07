"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions for 2000 candidate points.
    sigma : array of shape (2000,)
        GP standard deviation predictions for 2000 candidate points.
    f_best : float
        Best observed value so far.
    cost : array of shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = better candidate to evaluate).
    """
    # Deadline guard
    t_start = time.time()
    
    # Ensure inputs are arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Handle edge cases: ensure sigma and cost are positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    # Clip z to prevent extreme values that might cause numerical issues in norm.cdf/pdf
    # Although scipy is robust, clipping to [-10, 10] is a safe practice for stability
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Calculate Cost-Effective Expected Improvement (CEI)
    # CEI = EI / cost
    scores = ei / cost_safe
    
    # Ensure all values are finite. If any are not, replace with 0.
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Optional: If the deadline is approaching (unlikely for this vectorized op, but good practice)
    # We can just return. The operation is O(N) and very fast for N=2000.
    
    return scores
# EVOLVE-BLOCK-END
