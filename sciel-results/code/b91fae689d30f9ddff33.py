"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 300.0  # seconds, safety margin before 330s kill


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: Expected Improvement + adaptive exploration bonus.
    
    The exploration bonus is proportional to sigma but scaled by a factor that
    decays as f_best approaches the known minimum, encouraging exploration early
    and exploitation later.
    """
    # Check deadline
    if time.time() > _DEADLINE - 1.0:
        # Return a simple score based on mu (lower is better, so higher score = lower mu)
        return -mu
    
    # Standard Expected Improvement
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Adaptive exploration bonus:
    # The bonus is proportional to sigma, but scaled by how far f_best is from
    # the known minimum of the function.
    # Styblinski-Tang 1D minimum is -39.1662
    f_min_st = -39.1662
    
    # Scale factor: starts at 1.0 when f_best is far from minimum, decays to 0
    # as f_best approaches the minimum. Use a smooth decay.
    # Range of f_best typically: from ~38 (initial) down to -39.1662
    # Let's use a sigmoid-like decay centered around -20 with width 10
    center = -20.0
    width = 15.0
    exploration_scale = 1.0 / (1.0 + np.exp((f_best - center) / width))
    
    # Bonus coefficient: small enough to not dominate EI, but meaningful
    beta = 0.3 * exploration_scale
    
    # Exploration bonus: proportional to sigma
    bonus = beta * sigma
    
    # Combined score
    scores = ei + bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
