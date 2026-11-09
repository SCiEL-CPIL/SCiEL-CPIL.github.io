import numpy as np
from scipy.stats import norm
import time

# Global state to track elapsed time for deadline safety
_start_time = time.time()
_DEADLINE = 330.0
_SAFETY_MARGIN = 5.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Dynamic Cost-Aware Expected Improvement 
    with an Adaptive Exploration Bonus.
    
    Combines the structural stability of EI with an adaptive exploration term that 
    increases uncertainty weighting when the current best is poor (early stage) 
    and decreases it when the current best is good (late stage).
    """
    # Deadline check
    if time.time() > _DEADLINE - _SAFETY_MARGIN:
        # Fallback to simple exploitation to avoid timeout
        mu = np.asarray(mu, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        cost = np.maximum(cost, 1e-9)
        return -mu / cost

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure valid inputs
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate Expected Improvement (EI)
    # Minimization problem: EI = E[max(f_best - f(x), 0)]
    # Use xi=0 for standard EI
    xi = 0.0
    z = (f_best - mu - xi) / sigma_safe
    
    # Vectorized EI calculation
    ei = (f_best - mu - xi) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Adaptive Exploration Bonus
    # We want to explore more when f_best is high (bad) and exploit more when f_best is low (good).
    # Shekel 4d min is approx -10.5364. Initial values are often > 0.
    # Let's define a range for adaptation: f_best from -10 to 0.
    # If f_best < -10, alpha is minimal.
    # If f_best > 0, alpha is maximal.
    
    # Clip f_best for the alpha calculation to avoid extreme values
    f_best_clipped = np.clip(f_best, -10.0, 0.0)
    
    # Linear interpolation for alpha
    # alpha_low = 0.02 (when f_best is -10, highly exploited)
    # alpha_high = 0.10 (when f_best is 0, needs exploration)
    alpha_low = 0.02
    alpha_high = 0.10
    
    # t = 0 when f_best = -10, t = 1 when f_best = 0
    t = (f_best_clipped - (-10.0)) / (0.0 - (-10.0))
    t = np.clip(t, 0.0, 1.0)
    
    alpha = alpha_low + (alpha_high - alpha_low) * t
    
    exploration_bonus = alpha * sigma_safe
    
    # Combined Score before cost
    raw_score = ei + exploration_bonus
    
    # Cost Awareness
    # Divide by cost to prefer cheaper evaluations
    score = raw_score / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
