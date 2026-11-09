# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Adjusted Probability of Improvement.
    Maximizes P(mu < f_best) / cost + small uncertainty bonus.
    """
    t_start = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute z-score for Probability of Improvement (minimization)
    # P(mu < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -30, 30)
    
    # Probability of Improvement
    poi = norm.cdf(z)
    
    # Cost-adjusted PoI: prioritize high probability of success per unit cost
    cost_adjusted_poi = poi / cost_safe
    
    # Small uncertainty bonus to encourage exploration
    # Scaled to be a minor component
    uncertainty_bonus = 0.05 * sigma_safe / (np.max(sigma_safe) + 1e-10)
    
    # Combined score: higher is better
    scores = cost_adjusted_poi + uncertainty_bonus
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    elapsed = time.time() - t_start
    if elapsed > 280:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
