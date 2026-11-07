"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Compute a cost-aware acquisition score for 2000 candidate points.
    
    Uses a hybrid approach:
    1. Expected Improvement (EI) adjusted for cost
    2. Exploration bonus based on uncertainty (sigma)
    3. Deadline guard to prevent timeout
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions in objective units
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions in objective units
    f_best : float
        Best observed value so far (lowest, since we minimize)
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate: 1 + mean(x scaled to [0,1])
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate)
    """
    # Deadline guard: track elapsed time
    t0 = time.time()
    deadline = 295.0  # safety margin before 300s limit
    
    # Ensure inputs are float arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    
    # Clamp sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for EI
    # For minimization: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (for minimization)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Clamp EI to be non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-adjusted EI
    # Divide by cost to get improvement per unit cost
    cost_safe = np.maximum(cost, 1e-9)
    ei_per_cost = ei / cost_safe
    
    # Exploration bonus: reward points with high uncertainty
    # Scale by a factor that balances exploitation vs exploration
    # The scale should be relative to the magnitude of EI
    ei_scale = np.max(ei_per_cost) if np.max(ei_per_cost) > 1e-12 else 1.0
    exploration_bonus = 0.3 * (sigma_safe / (sigma_safe.max() + 1e-12)) * ei_scale
    
    # Combine: cost-adjusted EI + exploration bonus
    scores = ei_per_cost + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    if time.time() - t0 > deadline:
        # Return best result found so far
        return scores
    
    return scores
# EVOLVE-BLOCK-END
