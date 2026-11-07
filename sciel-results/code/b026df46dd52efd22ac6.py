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

# Structural change: Use a cost-aware acquisition that combines:
# 1. Expected Improvement (EI) for exploitation
# 2. Uncertainty-based exploration with cost-aware weighting
# 3. A cost-aware penalty that accounts for the total cost budget

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining EI and uncertainty.
    
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
    
    # Uncertainty-based exploration term
    # This encourages exploration in regions with high uncertainty
    exploration = sigma
    
    # Cost-aware weighting:
    # We want to balance the cost of evaluation with the potential improvement.
    # Use a cost-aware factor that reduces the weight of high-cost evaluations.
    # The cost is in the range [1, 2] since cost = 1 + mean(x_scaled) and x_scaled in [0, 1].
    # We use a cost-aware factor that is 1 for low cost and decreases for high cost.
    # cost_factor = 1 / (1 + 0.5 * (cost - 1))
    # This gives cost_factor = 1 when cost=1, and cost_factor = 0.67 when cost=2.
    cost_factor = 1.0 / (1.0 + 0.5 * (cost - 1.0))
    
    # Combine EI and exploration with cost-aware weighting
    # Weight the EI term more heavily as we get closer to the optimum
    # Weight the exploration term more heavily when uncertainty is high
    score = cost_factor * (0.8 * ei + 0.2 * exploration)
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
