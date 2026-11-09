import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # Safety margin before 330s limit

def acquisition(mu, sigma, f_best, cost):
    # Check deadline - if too close, return a simple acquisition to avoid timeout
    if time.time() - _start_time > _DEADLINE:
        # Fallback: simple EI/cost
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score
    delta = f_best - mu
    z = delta / sigma_safe
    
    # Expected Improvement (EI)
    ei = delta * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Upper Confidence Bound (UCB) - encourages exploration
    kappa = 2.0
    ucb = -mu + kappa * sigma_safe
    
    # Probability of Improvement (PI) - encourages exploitation
    pi = norm.cdf(z)
    
    # Normalize each component to [0, 1] range for blending
    # EI is already non-negative
    ei_norm = ei / (ei.max() + 1e-9) if ei.max() > 1e-9 else ei
    
    # UCB can be negative, so normalize
    ucb_min = ucb.min()
    ucb_max = ucb.max()
    ucb_range = ucb_max - ucb_min + 1e-9
    ucb_norm = (ucb - ucb_min) / ucb_range
    
    # PI is already in [0, 1]
    pi_norm = pi
    
    # Blend: 50% EI, 30% UCB, 20% PI
    blended = 0.5 * ei_norm + 0.3 * ucb_norm + 0.2 * pi_norm
    
    # Cost-aware normalization: divide by sqrt(cost) to reduce penalty on expensive points
    cost_safe = np.sqrt(cost + 1e-9)
    scores = blended / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
