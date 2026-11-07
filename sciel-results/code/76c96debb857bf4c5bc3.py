import numpy as np
from scipy.stats import norm
import time

# Global deadline guard: 330s wall-clock limit, use 300s as safety margin
_START_TIME = time.time()
_DEADLINE = _START_TIME + 300.0

def acquisition(mu, sigma, f_best):
    global _START_TIME, _DEADLINE
    
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a reasonable fallback: just EI
        mu = np.asarray(mu, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        sigma = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        ei = np.where(np.isfinite(ei), ei, 0.0)
        return ei
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Expected Improvement for minimization
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Upper Confidence Bound for minimization
    # Use adaptive beta based on the spread of sigma
    # Larger sigma spread suggests more uncertainty, so increase exploration
    sigma_mean = np.mean(sigma)
    sigma_std = np.std(sigma)
    
    # Adaptive beta: base 2.5, scaled by relative uncertainty
    if sigma_mean > 1e-9:
        rel_uncertainty = sigma_std / sigma_mean
        beta = 2.5 + 2.0 * rel_uncertainty
    else:
        beta = 2.5
    
    # Cap beta to avoid excessive exploration
    beta = np.clip(beta, 1.5, 5.0)
    
    ucb = -mu + beta * sigma
    
    # Normalize both to [0, 1] range for combination
    ei_min, ei_max = np.min(ei), np.max(ei)
    ucb_min, ucb_max = np.min(ucb), np.max(ucb)
    
    ei_range = ei_max - ei_min
    ucb_range = ucb_max - ucb_min
    
    if ei_range > 1e-10:
        ei_norm = (ei - ei_min) / ei_range
    else:
        ei_norm = np.zeros_like(ei)
    
    if ucb_range > 1e-10:
        ucb_norm = (ucb - ucb_min) / ucb_range
    else:
        ucb_norm = np.zeros_like(ucb)
    
    # Weighted combination: 60% EI, 40% UCB
    # EI tends to be more focused, UCB provides exploration
    score = 0.6 * ei_norm + 0.4 * ucb_norm
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
