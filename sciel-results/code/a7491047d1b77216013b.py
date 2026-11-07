"""Cost-aware acquisition for Bayesian optimization on powell 4d."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 295.0  # 5s safety margin before 300s limit

# Precompute constants
# x is in [-4, 5]^4, scaled to [0, 1]: (x - (-4)) / (5 - (-4)) = (x + 4) / 9
# Cost = 1 + mean(x_scaled) where x_scaled = (x + 4) / 9
# We don't have x directly, but cost is given.

# For the acquisition function, we use a cost-aware expected improvement.
# The key insight: we want to balance exploration (high uncertainty) and exploitation
# (low predicted value) while accounting for the cost of evaluation.

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Parameters:
    - mu: (2000,) array of GP mean predictions
    - sigma: (2000,) array of GP standard deviations
    - f_best: float, best observed value
    - cost: (2000,) array of evaluation costs
    
    Returns:
    - (2000,) array of acquisition scores (higher = more desirable)
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)) or not np.all(np.isfinite(cost)):
        # Return a valid fallback: use negative predicted value (exploit)
        return -mu
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-10)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-10)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware: divide by cost
    # But also add a small exploration bonus based on uncertainty
    # to avoid getting stuck in exploitation-only mode
    # The exploration bonus helps when sigma is large relative to (f_best - mu)
    
    # Use a hybrid: cost-aware EI plus a small uncertainty bonus
    # The bonus is scaled by 1/cost to maintain cost-awareness
    exploration_bonus = 0.1 * sigma / cost
    
    # Combine
    score = ei / cost + exploration_bonus
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
