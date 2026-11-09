"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_DEADLINE = None

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 320.0  # safety margin before 330s
    return _DEADLINE

def _time_left():
    return max(0.0, _get_deadline() - time.time())


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions
    - sigma: array of shape (2000,) - GP standard deviation predictions
    - f_best: scalar - best observed value so far
    - cost: array of shape (2000,) - evaluation costs
    
    Returns:
    - scores: array of shape (2000,) - acquisition scores (higher = better)
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Replace non-finite values with safe defaults
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-3)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Use a weighted combination of:
    # 1. Expected Improvement (EI) - exploitation
    # 2. Upper Confidence Bound (UCB) - exploration
    # 3. Cost normalization
    
    # EI component
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # UCB component (beta controls exploration-exploitation tradeoff)
    # Use a moderate beta that adapts to the uncertainty
    beta = 2.0
    ucb = mu + beta * sigma
    
    # Combine EI and UCB with weights
    # Higher weight on EI when we're confident, more UCB when uncertain
    # Normalize EI and UCB to comparable scales
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ucb_range = np.ptp(ucb) if np.ptp(ucb) > 0 else 1.0
    
    ei_norm = ei / ei_max
    ucb_norm = (ucb - np.min(ucb)) / ucb_range
    
    # Weighted combination
    weight_ei = 0.6
    weight_ucb = 0.4
    combined = weight_ei * ei_norm + weight_ucb * ucb_norm
    
    # Cost-aware normalization: divide by cost to prefer cheap evaluations
    scores = combined / cost
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
