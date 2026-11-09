# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses scale-normalized Expected Improvement with linear cost penalty
    and adaptive exploration bonus.
    Score = EI/scale - penalty * cost + bonus * sigma/scale
    """
    t_start = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute z-score for EI (minimization)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -30, 30)
    
    # Expected Improvement for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Scale EI relative to problem scale for better conditioning
    # Use a robust scale: max of |f_best| and mean EI, with floor
    ei_mean = np.mean(ei)
    scale = max(abs(f_best), ei_mean, 1e-6)
    ei_scaled = ei / scale
    
    # Adaptive cost penalty: stronger when budget is tight
    # Cost ranges from 1 to ~2 (1 + mean of two [0,1] values)
    # Use a penalty that scales with the remaining budget pressure
    # Higher penalty early to conserve budget, lower late to exploit
    # f_best close to 0 means we're near optimum, reduce penalty
    gap = max(f_best, 1e-6)
    # Penalty coefficient: starts around 0.6, decreases as we approach optimum
    penalty_coeff = 0.4 + 0.3 * np.sqrt(gap)
    cost_penalty = penalty_coeff * cost_safe
    
    # Adaptive exploration bonus: stronger when f_best is far from optimum
    # This helps escape local minima in the three-hump landscape
    exploration_coeff = 0.2 * np.sqrt(gap)
    exploration_bonus = exploration_coeff * sigma_safe / scale
    
    # Combined score: higher is better
    scores = ei_scaled - cost_penalty + exploration_bonus
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    elapsed = time.time() - t_start
    if elapsed > 280:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
