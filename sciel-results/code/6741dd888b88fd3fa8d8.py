import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_TIMEOUT = 325.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Expected Improvement per unit cost.
    Structural change: replaces Pareto-frontier filtering with a direct
    cost-normalized improvement metric.
    """
    # Deadline guard
    if time.time() - _START_TIME > _TIMEOUT:
        return -cost.copy()
    
    n = len(mu)
    
    # Compute Expected Improvement
    sigma_safe = np.where(sigma < 1e-9, 1e-9, sigma)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scoring: EI divided by cost to the power alpha
    # This directly optimizes improvement per unit cost
    # alpha > 1 penalizes high-cost evaluations more heavily
    alpha = 1.5
    cost_safe = np.where(cost < 1e-9, 1e-9, cost)
    scores = ei / (cost_safe ** alpha)
    
    # Add a small exploration bonus for high-uncertainty regions
    # This helps escape local exploitation and encourages diverse sampling
    # Weighted by 1/cost so exploration also favors cheaper points
    exploration_weight = 0.1
    exploration = exploration_weight * sigma_safe / cost_safe
    scores = scores + exploration
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
