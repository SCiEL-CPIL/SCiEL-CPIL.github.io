import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290  # 300 seconds limit per evaluation, keep safety margin

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (PoI)
    normalized by cost, with a structural guard against numerical issues.
    
    The strategy uses PoI because it is computationally efficient and robust.
    Normalizing by cost encourages the optimizer to prefer cheaper evaluations,
    which is crucial given the cost model 1 + mean(x_scaled).
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: return 1/cost to at least prefer cheap points
        cost_safe = np.maximum(np.asarray(cost), 1e-9)
        return 1.0 / cost_safe

    # Convert to arrays and sanitize
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Ensure sigma is strictly positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate z-score for Probability of Improvement
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    
    # PoI is the CDF of the standard normal at z
    poi = norm.cdf(z)
    
    # Cost-aware score: PoI per unit cost
    # Higher score indicates a more desirable point to evaluate next
    scores = poi / cost_safe
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=0.0)
    
    return scores
