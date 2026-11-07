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
    
    # Compute UCB with adaptive exploration
    # Use a beta that adapts based on the spread of sigma
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma) + 1e-9
    
    # Adaptive beta: higher when uncertainty is more variable
    # Start with moderate exploration, adjust based on sigma distribution
    beta = 1.5 + 0.5 * (sigma_std / sigma_mean)
    
    # UCB score
    ucb = mu + beta * sigma
    
    # For minimization, we want low mu and high sigma (uncertainty)
    # Transform: high score = low mu + high sigma
    # ucb_min = mu - beta * sigma (this is lower for uncertain low regions)
    # We want to maximize, so negate: -ucb_min = -mu + beta * sigma
    ucb_score = -mu + beta * sigma
    
    # Normalize both EI and UCB to comparable scales
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
        ucb_norm = np.zeros_like(ucb_score)
    
    # Hybrid: blend EI and UCB
    # Weight shifts toward exploitation as we find better solutions
    # Use the ratio of f_best to typical values to adjust
    blend_weight = 0.5  # Start balanced
    
    hybrid = blend_weight * ei_norm + (1 - blend_weight) * ucb_norm
    
    # Cost-aware: divide by cost
    # cost is "1 + mean(x scaled to [0, 1])"
    # Lower cost is better, so divide by cost to get score per unit cost
    scores = hybrid / cost
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores

# EVOLVE-BLOCK-END
