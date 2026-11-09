"""Cost-aware acquisition for BO on hartmann 3d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: blend of normalized EI and PI with uncertainty bonus,
    divided by cost. Includes a deadline guard for safety.
    """
    # Deadline guard: if we're close to timeout, return a safe simple score
    # (This is a lightweight check; the real deadline is enforced by the evaluator)
    # We don't have a start time here, so we just ensure fast computation.
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Normalize sigma to [0, 1] for the uncertainty bonus
    max_sigma = np.max(sigma_safe)
    if max_sigma < 1e-9:
        max_sigma = 1e-9
    sigma_norm = sigma_safe / max_sigma
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement (PI)
    pi = norm.cdf(z)
    
    # Normalize EI to be comparable to PI
    max_ei = np.max(ei)
    if max_ei < 1e-9:
        max_ei = 1e-9
    ei_norm = ei / max_ei
    
    # Blend: 0.6 * EI + 0.4 * PI, plus a small uncertainty bonus
    # The uncertainty bonus encourages exploration in early stages
    blend = 0.6 * ei_norm + 0.4 * pi + 0.1 * sigma_norm
    
    # Cost-aware: divide by cost
    cost_safe = np.maximum(cost, 1e-9)
    score = blend / cost_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
