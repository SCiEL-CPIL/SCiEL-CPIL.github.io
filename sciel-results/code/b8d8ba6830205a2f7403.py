"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 295  # 295 seconds limit (5s safety margin before 300s kill)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Probability of Improvement (PoI) with sharpened confidence and relaxed cost penalty.
    
    This function computes a modified Probability of Improvement for each candidate.
    The PoI is sharpened by dividing the z-score by a temperature factor (T=0.8) to increase
    the contrast between high and low confidence predictions. The cost normalization is
    relaxed (cost^-0.8) to slightly reduce the penalty for evaluating points in higher-cost
    regions, which may contain the global minimum.
    
    Args:
        mu: array of shape (2000,) - GP mean
        sigma: array of shape (2000,) - GP standard deviation
        f_best: float - best observed value
        cost: array of shape (2000,) - cost of evaluating each point
        
    Returns:
        scores: array of shape (2000,) - acquisition scores (higher is better)
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: return 1/cost to prefer cheap points
        cost_safe = np.maximum(cost, 1e-9)
        return 1.0 / cost_safe
    
    # Ensure inputs are arrays and handle types
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate z-score for Probability of Improvement
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid extreme values causing numerical issues in CDF
    z_clipped = np.clip(z, -10, 10)
    
    # Sharpen the PoI by scaling the z-score (T < 1 makes the CDF steeper)
    # This increases the weight of high-confidence improvements
    temperature = 0.8
    z_sharp = z_clipped / temperature
    
    # PoI is the CDF of the standard normal at the sharpened z
    poi = norm.cdf(z_sharp)
    
    # Cost-aware scoring: PoI per unit cost, with relaxed penalty
    # Using cost^-0.8 instead of cost^-1 reduces the penalty for expensive points
    scores = poi / (cost_safe ** 0.8)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
