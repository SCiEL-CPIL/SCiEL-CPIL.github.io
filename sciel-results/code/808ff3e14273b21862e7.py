"""Cost-aware acquisition for rastrigin 2d with explicit deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Module-level deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0  # 300s limit, 20s safety margin


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Expected Improvement with exploration bonus.
    
    Parameters:
        mu: (2000,) array of GP mean predictions
        sigma: (2000,) array of GP std predictions
        f_best: scalar, best observed value
        cost: (2000,) array of evaluation costs (1 + mean(x_scaled))
    
    Returns:
        (2000,) array of acquisition scores (higher = better)
    """
    # Deadline guard: if we're running out of time, return a simple score
    if time.time() > _DEADLINE:
        # Fallback: just use -mu/cost (greedy)
        return -mu / np.maximum(cost, 1e-9)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement with small exploration offset xi
    xi = 0.01
    improvement = f_best - mu - xi
    z = improvement / sigma_safe
    
    # EI = improvement * Phi(z) + sigma * phi(z)
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add exploration bonus proportional to sigma to help escape local minima
    # This encourages sampling uncertain regions
    exploration_bonus = 0.1 * sigma_safe
    
    # Combine EI and exploration, then divide by cost for cost-awareness
    # Using a slightly different weighting: emphasize EI more but still explore
    score = (ei + exploration_bonus) / np.maximum(cost, 1e-9)
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
