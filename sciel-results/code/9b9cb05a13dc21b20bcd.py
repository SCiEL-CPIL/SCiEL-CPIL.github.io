# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Probability of Improvement (PoI) normalized by cost, plus
    a GP uncertainty bonus to balance exploitation and exploration.
    Score = PoI / cost + bonus * sigma / scale
    """
    # Deadline guard
    t_start = time.time()
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero and negative sigma
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute z-score for PoI
    # For minimization: P(mu < f_best) = Phi((f_best - mu) / sigma)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Clip z to avoid numerical issues
    z = np.clip(z, -30, 30)
    
    # Compute PoI (Probability of Improvement for minimization)
    poi = norm.cdf(z)
    
    # Ensure PoI is in [0, 1]
    poi = np.clip(poi, 0.0, 1.0)
    
    # Cost-normalized PoI
    poi_cost = poi / cost_safe
    
    # Add a small exploration bonus based on uncertainty
    # This helps explore high-uncertainty regions even when PoI is low
    # The bonus is scaled by sigma relative to the current best
    scale = max(abs(f_best), 1e-6)
    exploration_bonus = 0.15 * sigma_safe / scale
    
    # Combined score: higher is better
    scores = poi_cost + exploration_bonus
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - t_start
    if elapsed > 280:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
