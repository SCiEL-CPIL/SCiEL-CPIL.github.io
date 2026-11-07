"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement (EI)
    with a penalty for high-cost regions and a small exploration bonus.
    
    mu, sigma, f_best, cost: arrays of shape (2000,)
    Returns: array of 2000 finite scores (higher = better candidate to evaluate)
    """
    # Deadline guard: if we've spent too much time, return a simple fallback
    # This is a safety net; the main logic below should be fast.
    # We don't have a global timer here, but we can check if inputs are valid.
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Cost-awareness: divide by cost to prefer cheaper evaluations
    # But we also want to ensure we don't just pick the cheapest point repeatedly.
    # A common approach is EI / (cost + epsilon) or EI * (1 / cost^alpha).
    # Let's use a soft penalty: EI / (cost * (1 + 0.5 * cost))
    # Actually, the seed program used EI / cost. Let's try a slightly different weighting.
    # If cost is low, EI/cost is large. If cost is high, it's small.
    # But sometimes we need to explore high-cost regions if EI is very high.
    
    # Let's use a hybrid: EI / cost + small exploration term based on sigma
    # The exploration term encourages visiting uncertain regions.
    # sigma is already in the EI term, but let's add a direct sigma bonus.
    
    # Normalize cost to [0, 1] for a smoother penalty?
    # cost is in [1, 2] roughly (1 + mean(x_scaled)).
    # Let's stick with EI / cost but add a small sigma bonus to encourage exploration.
    
    # To avoid numerical issues, clip EI to be non-negative (it should be, but just in case)
    ei = np.maximum(ei, 0)
    
    # Score = EI / cost + beta * sigma
    # beta controls the exploration/exploitation trade-off.
    # A small beta ensures we don't just pick the most uncertain point.
    beta = 0.01
    
    scores = ei / cost + beta * sigma
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
