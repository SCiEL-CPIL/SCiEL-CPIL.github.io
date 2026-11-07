"""Cost-aware acquisition function for Bayesian optimization (ackley 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 325.0  # Safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Expected Improvement with adaptive cost penalty.
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        sigma_safe = np.maximum(sigma, 1e-8)
        cost_safe = np.maximum(cost, 1e-6)
        fallback = sigma_safe / cost_safe
        return np.where(np.isfinite(fallback), fallback, 0.0)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Expected Improvement (EI)
    diff = f_best - mu
    z = diff / sigma_safe
    z_clipped = np.clip(z, -15, 15)
    
    cdf = norm.cdf(z_clipped)
    pdf = norm.pdf(z_clipped)
    
    ei = diff * cdf + sigma_safe * pdf
    ei = np.maximum(ei, 0.0)
    
    # Adaptive cost penalty based on how close we are to the minimum
    # The minimum of Ackley is 0, so we can gauge progress by f_best
    # When f_best is close to 0, we should be more conservative with cost
    # When f_best is far from 0, we can afford to explore more expensive regions
    
    # Normalize f_best: assume reasonable range for Ackley 2D is [0, ~10]
    # Use a soft normalization
    f_best_clipped = np.clip(f_best, 0.0, 10.0)
    progress = 1.0 - f_best_clipped / 10.0  # 0 when f_best=10, 1 when f_best=0
    
    # Cost penalty strength: stronger when we're close to the optimum
    # Early on (low progress), use weaker penalty to allow exploration
    # Later (high progress), use stronger penalty to save budget
    penalty_strength = 0.3 + 0.5 * progress  # Range: [0.3, 0.8]
    
    # Soft cost penalty
    cost_normalized = np.clip(cost - 1.0, 0.0, 1.0)  # Range: [0, 1]
    cost_penalty = 1.0 + penalty_strength * cost_normalized
    cost_penalty = np.maximum(cost_penalty, 1e-6)
    
    score = ei / cost_penalty
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
