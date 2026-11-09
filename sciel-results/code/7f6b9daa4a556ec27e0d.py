import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a blend of Normalized EI and PI
    with a dynamic exploration bonus based on uncertainty.
    """
    t_start = time.time()
    
    # Safety for sigma
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Normalize to [0, 1] using Min-Max
    ei_min, ei_max = np.min(ei), np.max(ei)
    pi_min, pi_max = np.min(pi), np.max(pi)
    
    ei_range = ei_max - ei_min + 1e-9
    pi_range = pi_max - pi_min + 1e-9
    
    ei_norm = (ei - ei_min) / ei_range
    pi_norm = (pi - pi_min) / pi_range
    
    # Dynamic exploration bonus: scale sigma relative to max sigma
    max_sigma = np.max(sigma_safe) + 1e-9
    sigma_rel = sigma_safe / max_sigma
    
    # Blend: weight EI and PI, add exploration bonus
    # As sigma decreases (GP becomes more confident), exploration bonus shrinks
    blended = 0.5 * ei_norm + 0.3 * pi_norm + 0.2 * sigma_rel
    
    # Cost-aware adjustment
    cost_safe = np.maximum(cost, 1e-10)
    scores = blended / cost_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Time guard
    t_elapsed = time.time() - t_start
    if t_elapsed > 280:
        return scores
        
    return scores
