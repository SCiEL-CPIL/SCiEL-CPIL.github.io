"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # 300s limit, keep 10s safety margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions at 2000 candidate points
    sigma : array-like, shape (2000,)
        GP standard deviation predictions at 2000 candidate points
    f_best : float
        Best observed function value so far
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate next)
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return a simple fallback to avoid timeout
        cost_safe = np.maximum(cost, 1e-9)
        return np.ones(2000) / cost_safe
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Compute z-scores
    z = (f_best - mu) / sigma
    
    # Clip z to avoid numerical issues with extreme values
    z_clipped = np.clip(z, -50.0, 50.0)
    
    # Expected Improvement (EI) for minimization
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement (PoI)
    poi = norm.cdf(z_clipped)
    
    # Lower Confidence Bound (LCB) for minimization
    # LCB = mu - beta * sigma; we maximize -LCB = -mu + beta * sigma
    beta = 1.5
    lcb = -mu + beta * sigma
    
    # Blend EI and LCB for robustness
    # EI focuses on improvement, LCB balances exploration/exploitation
    blend = 0.7 * ei + 0.3 * lcb
    
    # Add a small exploration bonus based on uncertainty
    # This helps steer evaluations toward under-explored regions
    exploration_bonus = 0.05 * sigma / (sigma + 1.0)
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    # The cost model is 1 + mean(x_scaled), so cost in [1, 2]
    scores = (blend + exploration_bonus) / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
