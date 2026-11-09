import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    
    # Initialize start time on first call
    if _start_time is None:
        _start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Check if we're approaching the deadline (330s limit, use 320s as safety margin)
    elapsed = time.time() - _start_time
    if elapsed > 320:
        safe_cost = np.maximum(cost, 1e-10)
        return -mu / safe_cost
    
    # Small epsilon to prevent division by zero
    eps = 1e-12
    safe_sigma = np.maximum(sigma, eps)
    safe_cost = np.maximum(cost, eps)
    
    # Calculate z-score for minimization
    z = (f_best - mu) / safe_sigma
    z_clipped = np.clip(z, -10, 10)
    
    # Probability of Improvement (PI)
    pi_val = norm.cdf(z_clipped)
    
    # Expected Improvement (EI)
    pdf_val = norm.pdf(z_clipped)
    ei_val = (f_best - mu) * pi_val + safe_sigma * pdf_val
    
    # Upper Confidence Bound (UCB) for minimization
    # UCB favors points with low mean and high uncertainty
    # beta controls the exploration-exploitation tradeoff
    # Using a dynamic beta that decreases as we get closer to the optimum
    beta = 0.5
    
    ucb_val = mu - beta * safe_sigma
    
    # Normalize UCB to be comparable to PI and EI
    # For minimization, lower UCB is better, so we negate it to make higher scores better
    # Scale by f_best to make it scale-invariant
    scale_factor = max(abs(f_best), np.mean(safe_sigma) + eps)
    ucb_norm = -ucb_val / scale_factor
    
    # Structural Change: Use a different acquisition function family
    # Instead of blending PI and EI, use a combination that emphasizes
    # the expected improvement relative to the current best while
    # accounting for cost more aggressively
    # 
    # New approach: Use a cost-aware variant of Expected Improvement
    # with a different weighting that emphasizes points with high
    # probability of being significantly better than current best
    #
    # Weighted EI: Multiply EI by a factor that increases with
    # the z-score to emphasize points that are likely to be much better
    z_positive = np.maximum(z_clipped, 0)
    weight_factor = 1.0 + 0.5 * z_positive
    
    weighted_ei = ei_val * weight_factor
    
    # Normalize weighted EI
    weighted_ei_norm = weighted_ei / scale_factor
    
    # Combine weighted EI with PI using different weights
    # Emphasize weighted EI more as it captures the magnitude of improvement
    combined_score = 0.3 * pi_val + 0.7 * weighted_ei_norm
    
    # Cost-aware: Use a more aggressive cost penalty
    # Instead of 1/cost, use 1/cost^alpha with alpha > 1 to
    # penalize expensive points more heavily
    alpha = 1.5
    cost_penalty = safe_cost ** alpha
    score = combined_score / cost_penalty
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
