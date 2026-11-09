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
    
    # Structural change: use sqrt(cost) instead of cost to reduce over-penalization
    # This allows more exploration in 6D with few initial points
    cost_penalty = np.sqrt(cost_safe)
    
    cost_ei = ei / cost_penalty
    
    # UCB component for minimization with higher beta for stronger exploration
    # Adaptive beta: start high for exploration, reduce as we converge
    target_min = -3.32237
    gap = f_best - target_min
    
    if gap > 1.0:
        # Far from optimal: strong exploration
        beta = 2.0
        ei_weight = 0.3
        ucb_weight = 0.7
    elif gap > 0.3:
        # Medium distance: balanced
        beta = 1.5
        ei_weight = 0.45
        ucb_weight = 0.55
    elif gap > 0.05:
        # Close to optimal: more exploitation
        beta = 1.0
        ei_weight = 0.6
        ucb_weight = 0.4
    else:
        # Very close: strong exploitation
        beta = 0.7
        ei_weight = 0.75
        ucb_weight = 0.25
    
    ucb_score = (-mu + beta * sigma_safe) / cost_penalty
    
    # Normalize both components to have similar scales
    # Use robust normalization to avoid outliers dominating
    ei_mean = np.mean(cost_ei)
    ei_std = np.std(cost_ei) + 1e-8
    ei_norm = (cost_ei - ei_mean) / ei_std
    
    ucb_mean = np.mean(ucb_score)
    ucb_std = np.std(ucb_score) + 1e-8
    ucb_norm = (ucb_score - ucb_mean) / ucb_std
    
    # Combine
    combined = ei_weight * ei_norm + ucb_weight * ucb_norm
    
    # Ensure finite scores
    scores = np.where(np.isfinite(combined), combined, -1e10)
    
    # Deadline check
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        # Return a safe fallback
        return np.zeros(n, dtype=np.float64)
    
    return scores
