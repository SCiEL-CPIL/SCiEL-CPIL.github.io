"""Cost-aware acquisition for BO on three-hump camel 2d with shifted instances."""
import numpy as np
from scipy.stats import norm
import time


# Module-level deadline guard
_START_TIME = time.time()
# Leave a safety margin of 30 seconds before the 330s kill
_DEADLINE = _START_TIME + 295.0


def _check_deadline():
    """Return True if we should stop to avoid the deadline."""
    return time.time() > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement and
    a sigma-based exploration bonus, normalized by cost.
    
    Parameters:
        mu: (2000,) array of GP mean predictions in objective units
        sigma: (2000,) array of GP std dev predictions in objective units
        f_best: scalar, best observed value so far
        cost: (2000,) array of evaluation costs (1 + mean(x_scaled_to_[0,1]))
    
    Returns:
        (2000,) array of acquisition scores (higher = evaluate next)
    """
    # Check deadline - if we're close, just return a safe fallback
    if _check_deadline():
        # Return something finite and reasonable: prefer low mu (exploitation)
        # with minimal computation
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)
    
    # Ensure inputs are proper
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Standard Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Add exploration bonus proportional to sigma
    # This encourages visiting uncertain regions to escape local optima
    # The beta coefficient balances exploration vs exploitation
    beta = 0.1
    exploration_bonus = beta * sigma_safe
    
    # Combine EI and exploration bonus, then normalize by cost
    score = (ei + exploration_bonus) / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
