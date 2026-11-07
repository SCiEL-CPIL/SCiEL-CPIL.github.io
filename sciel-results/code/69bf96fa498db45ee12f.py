"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline tracking
_START_TIME = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Probability of Improvement (POI)
    with a cost-aware modification.
    
    POI = P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    
    We modify it to account for evaluation cost:
    Score = POI / (cost^delta) * (1 + sigma^gamma)
    
    - POI provides a sharp, decisive selection signal, committing to promising
      low-cost regions faster and avoiding dilution of exploration.
    - The (1 + sigma^gamma) term adds a mild exploration bonus for high-uncertainty
      regions, preventing premature convergence.
    - cost^delta with delta < 1 softens the cost penalty, allowing the algorithm
      to consider expensive points if the potential gain is high enough.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Deadline guard: return safe fallback if we're running out of time
    elapsed = time.time() - _START_TIME
    if elapsed > 315.0:
        # Fallback: simple POI normalized by cost
        sigma_arr = np.asarray(sigma, dtype=np.float64)
        cost_arr = np.asarray(cost, dtype=np.float64)
        z = (f_best - mu) / np.maximum(sigma_arr, 1e-10)
        poi = norm.cdf(z)
        fallback = poi / np.maximum(cost_arr, 1e-10)
        return np.where(np.isfinite(fallback), fallback, 0.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    POI = norm.cdf(z)
    
    # Parameters
    gamma = 0.5  # Mild exploration bonus
    delta = 0.5  # Soften cost penalty
    
    # Exploration bonus: 1 + sigma^gamma
    exploration_bonus = 1.0 + sigma_safe ** gamma
    
    # Cost term: 1 / cost^delta
    cost_term = 1.0 / (cost_safe ** delta)
    
    # Final score
    scores = POI * exploration_bonus * cost_term
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
