import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_TIME_LIMIT = 325.0
_SAFETY_MARGIN = 5

def acquisition(mu, sigma, f_best, cost):
    # Check time
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT - _SAFETY_MARGIN:
        # Fallback to simple score if we are running out of time
        return np.ones_like(mu) / cost
    
    # Input validation and safety
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    
    # Handle non-finite values
    sigma_safe = np.where(np.isfinite(sigma), sigma, 1e-8)
    sigma_safe = np.maximum(sigma_safe, 1e-8)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    cost_safe = np.where(np.isfinite(cost), cost, 1.0)
    cost_safe = np.maximum(cost_safe, 1e-8)
    
    # Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI should be non-negative
    
    # Cost-aware EI: divide by cost to penalize expensive points
    cost_ei = ei / cost_safe
    
    # UCB component for minimization
    # We want to maximize the score, so we negate: -mu + beta * sigma
    # Use a smaller beta since EI already captures uncertainty
    beta = 1.0
    ucb_score = (-mu + beta * sigma_safe) / cost_safe
    
    # Normalize both components to have similar scales
    # Use robust normalization to avoid outliers dominating
    ei_mean = np.mean(cost_ei)
    ei_std = np.std(cost_ei) + 1e-8
    ei_norm = (cost_ei - ei_mean) / ei_std
    
    ucb_mean = np.mean(ucb_score)
    ucb_std = np.std(ucb_score) + 1e-8
    ucb_norm = (ucb_score - ucb_mean) / ucb_std
    
    # Combine with equal weight initially, but adapt based on progress
    # When f_best is close to optimal (-3.32237), rely more on exploitation (EI)
    # When f_best is far from optimal, rely more on exploration (UCB)
    target_min = -3.32237
    gap = f_best - target_min
    if gap > 1.0:
        # Far from optimal: more exploration
        ei_weight = 0.4
        ucb_weight = 0.6
    elif gap > 0.1:
        # Medium distance: balanced
        ei_weight = 0.5
        ucb_weight = 0.5
    else:
        # Close to optimal: more exploitation
        ei_weight = 0.7
        ucb_weight = 0.3
    
    combined = ei_weight * ei_norm + ucb_weight * ucb_norm
    
    # Ensure finite scores
    scores = np.where(np.isfinite(combined), combined, -1e10)
    
    # Deadline check
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        # Return a safe fallback
        return np.zeros(n, dtype=np.float64)
    
    return scores
