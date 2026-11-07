"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    with an exploration bonus, normalized by cost.
    """
    # Deadline guard: if we're running out of time, return a safe fallback
    # (this is a heuristic; the actual timeout is managed by the evaluator)
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement: E[max(f_best - f(x), 0)]
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Exploration bonus: encourage points with high uncertainty
    # Use a small multiple of sigma to balance exploitation and exploration
    exploration_bonus = 0.1 * sigma_safe
    
    # Combine EI and exploration
    acquisition_score = ei + exploration_bonus
    
    # Cost-aware: divide by cost (higher cost = lower priority)
    # Cost is in [1, 2]
    score = acquisition_score / cost
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
    # EVOLVE-BLOCK-END
