"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 290.0  # 10s safety margin

def _check_deadline():
    return time.time() - _start_time > _TIMEOUT

def acquisition(mu, sigma, f_best, cost):
    # Check time
    if _check_deadline():
        # Fallback: return finite scores
        return np.ones_like(mu)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Compute PI (Probability of Improvement)
    pi = norm.cdf(z)
    pi = np.nan_to_num(pi, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Compute UCB with adaptive exploration
    # Use a beta that adapts based on the spread of sigma
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma) + 1e-9
    
    # Adaptive beta: higher when uncertainty is more variable
    beta = 1.5 + 0.5 * (sigma_std / sigma_mean)
    
    # UCB score for minimization: high score = low mu + high sigma
    ucb_score = -mu + beta * sigma
    
    # Normalize components to comparable scales
    ei_max = np.max(ei)
    if ei_max > 1e-9:
        ei_norm = ei / ei_max
    else:
        ei_norm = ei
    
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    ucb_range = ucb_max - ucb_min
    if ucb_range > 1e-9:
        ucb_norm = (ucb_score - ucb_min) / ucb_range
    else:
        ucb_norm = np.zeros_like(2000)
    
    pi_max = np.max(pi)
    if pi_max > 1e-9:
        pi_norm = pi / pi_max
    else:
        pi_norm = pi
    
    # Structural change: Use a product-based acquisition instead of linear blend
    # Multiply EI by PI to reward regions with both high expected improvement
    # and high probability of improvement, then combine with UCB
    # This creates a more selective search that focuses on promising uncertain regions
    
    # The product ei_norm * pi_norm emphasizes regions where both metrics agree
    product_score = ei_norm * pi_norm
    
    # Blend product score with UCB using a fixed weight
    # Product score favors exploitation in high-confidence improvement regions
    # UCB favors exploration in uncertain regions
    blend_weight = 0.4  # Weight for product score
    
    hybrid = blend_weight * product_score + (1 - blend_weight) * ucb_norm
    
    # Cost-aware: divide by cost
    scores = hybrid / cost
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores

# EVOLVE-BLOCK-END
