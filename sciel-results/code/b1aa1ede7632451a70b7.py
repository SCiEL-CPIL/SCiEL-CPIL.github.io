# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Probability of Improvement (PI) acquisition function.
    
    This structural change shifts from Expected Improvement (EI) to Probability
    of Improvement (PI), normalized by the evaluation cost. This aligns the
    search with the cost-aware objective, favoring points that are both likely
    to improve upon the current best and inexpensive to evaluate.
    """
    # Ensure numerical stability
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Calculate z-score for Probability of Improvement
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid extreme values
    z_clipped = np.clip(z, -10, 10)
    
    # Probability of Improvement: P(f(x) < f_best) = Phi(z)
    pi = norm.cdf(z_clipped)
    
    # Ensure cost is positive to avoid division by zero or negative costs
    cost_safe = np.maximum(cost, eps)
    
    # Cost-aware score: PI / cost
    # This penalizes expensive points, directing the search towards
    # cheaper regions that still offer a good chance of improvement.
    scores = pi / cost_safe
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
