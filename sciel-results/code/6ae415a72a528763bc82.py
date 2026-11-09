"""Cost-aware acquisition function using normalized EI and UCB with deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_DEADLINE_START = time.time()
_HARD_LIMIT = 330.0 - 60.0


def acquisition(mu, sigma, f_best, cost):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    if (time.time() - _DEADLINE_START) > _HARD_LIMIT:
        return np.nan_to_num(-mu / np.maximum(cost, 1e-6), nan=0.0, posinf=0.0, neginf=0.0)

    cost_safe = np.maximum(cost, 1e-6)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    z = np.clip((f_best - mu) / sigma_safe, -1e4, 1e4)
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    ei = (f_best - mu) * cdf + sigma * pdf
    ei = np.maximum(ei, 0.0)
    
    # Normalize EI and Sigma to [0, 1] range to balance their scales
    ei_max = np.max(ei)
    sigma_max = np.max(sigma)
    
    if ei_max > 1e-9:
        ei_norm = ei / ei_max
    else:
        ei_norm = ei
        
    if sigma_max > 1e-9:
        sigma_norm = sigma / sigma_max
    else:
        sigma_norm = sigma
        
    # Blend: 70% exploitation (EI), 30% exploration (Sigma)
    # Then divide by cost to penalize expensive points
    alpha = 0.3
    blended = (1 - alpha) * ei_norm + alpha * sigma_norm
    
    score = blended / cost_safe
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
