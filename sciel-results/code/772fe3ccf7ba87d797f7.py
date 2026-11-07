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
    
    # Adaptive beta: start high for exploration, decrease as we converge
    # Use fraction of points below f_best as convergence indicator
    frac_better = np.mean(mu < f_best)
    
    # Beta decreases smoothly from ~2.0 (early) to ~0.8 (late)
    beta = 2.0 - 1.2 * frac_better
    beta = np.clip(beta, 0.5, 2.5)
    
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
    
    # Weight EI more as we converge (smoother transition)
    # Early: more UCB for exploration; Late: more EI for exploitation
    weight_ei = 0.3 + 0.6 * frac_better
    weight_ucb = 1.0 - weight_ei
    
    score = weight_ei * ei_norm + weight_ucb * ucb_norm
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
