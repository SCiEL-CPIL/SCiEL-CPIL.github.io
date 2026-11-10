import numpy as np
from scipy.stats import norm
import time

# Global time tracking for deadline guard
_start_time = time.time()
_TIME_LIMIT = 300.0  # Safety margin before 330s

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    and adaptive Upper Confidence Bound (UCB), normalized by evaluation cost.
    
    The UCB exploration weight adapts to the local cost structure:
    - In cheap regions, we allow more exploration (higher beta)
    - In expensive regions, we favor exploitation (lower beta)
    This creates a cost-aware balance between exploration and exploitation.
    """
    # Check deadline
    if time.time() - _start_time > _TIME_LIMIT:
        # Return a safe, fast score to finish quickly
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        fallback = ei / cost_safe
        return np.where(np.isfinite(fallback), fallback, 0.0)

    # Ensure sigma and cost are positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-aware adaptive UCB
    # Compute normalized cost in [0, 1] for each candidate
    cost_min = np.min(cost_safe)
    cost_max = np.max(cost_safe)
    cost_range = cost_max - cost_min
    
    if cost_range < 1e-12:
        cost_norm = np.zeros_like(cost_safe)
    else:
        cost_norm = (cost_safe - cost_min) / cost_range
    
    # Adaptive exploration weight:
    # In cheap regions (cost_norm near 0), use higher beta (more exploration)
    # In expensive regions (cost_norm near 1), use lower beta (more exploitation)
    # beta ranges from beta_max (cheap) to beta_min (expensive)
    beta_min = 0.2   # Minimum exploration weight in expensive regions
    beta_max = 0.8   # Maximum exploration weight in cheap regions
    beta = beta_max - (beta_max - beta_min) * cost_norm
    
    # UCB for minimization: -mu + beta * sigma
    ucb = -mu + beta * sigma_safe
    
    # Combine EI and UCB
    # EI is the primary signal (exploitation), UCB provides exploration
    # We weight them to balance: EI dominates when it's strong, UCB helps in uncertain regions
    score_raw = ei + 0.3 * ucb
    
    # Cost-aware normalization: divide by cost to penalize expensive evaluations
    final_scores = score_raw / cost_safe
    
    # Ensure finite scores
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return final_scores
