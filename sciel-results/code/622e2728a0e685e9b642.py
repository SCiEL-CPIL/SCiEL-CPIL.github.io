# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = time.time()
_DEADLINE = 325.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        # Fallback: return UCB-like scores quickly
        sigma_safe = np.maximum(sigma, 1e-9)
        return -mu + 1.5 * sigma_safe
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute UCB for minimization
    # For minimization, we want low mu, so -mu is high when mu is low
    # UCB: -mu + beta * sigma
    
    # Adaptive strategy: balance exploration and exploitation
    # Use the distribution of sigma to determine exploration level
    mean_sigma = np.mean(sigma_safe)
    std_sigma = np.std(sigma_safe)
    
    # Adaptive beta based on uncertainty
    if mean_sigma > 0:
        # Higher beta when uncertainty is high relative to mean
        beta = 1.0 + 0.8 * (std_sigma / (mean_sigma + 1e-9))
        beta = np.clip(beta, 0.5, 2.5)
    else:
        beta = 1.0
    
    # UCB score for minimization
    ucb = -mu + beta * sigma_safe
    
    # Normalize both to [0, 1] for combining
    def normalize(arr):
        mn, mx = np.min(arr), np.max(arr)
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    ei_norm = normalize(ei)
    ucb_norm = normalize(ucb)
    
    # Combine based on convergence indicator
    # frac_better: fraction of points where mu < f_best
    # As this increases, we're finding better points, so exploit more
    frac_better = np.mean(mu < f_best)
    
    # Weight EI more as we converge
    # Use a smoother transition
    weight_ei = 0.4 + 0.5 * frac_better
    weight_ucb = 1.0 - weight_ei
    
    score = weight_ei * ei_norm + weight_ucb * ucb_norm
    
    # Add a small exploration bonus for points with high sigma
    # This helps escape local minima in the acquisition landscape
    sigma_rank = np.argsort(np.argsort(sigma_safe)) / n
    score += 0.1 * sigma_rank
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
