"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Computes a diversity-aware, cost-aware acquisition function.
    
    Combines Expected Improvement (EI) with a diversity penalty derived from
    the GP's predictive variance (sigma). Regions with low sigma are considered
    'crowded' or well-explored, so candidates there receive a penalty to
    encourage exploration of new areas.
    
    Args:
        mu: array of shape (N,) predicted mean
        sigma: array of shape (N,) predicted std deviation
        f_best: float, best value found so far
        cost: array of shape (N,) evaluation cost for each candidate
        
    Returns:
        scores: array of shape (N,) acquisition values (higher is better)
    """
    # Deadline guard: if we've been running too long, return a simple, fast score
    if not hasattr(acquisition, '_start_time'):
        acquisition._start_time = time.time()
    elapsed = time.time() - acquisition._start_time
    if elapsed > 280:
        # Fallback: simple EI per cost, very fast
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to a small positive value to prevent division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Clip cost to a small positive value
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate PDF and CDF
    pdf = norm.pdf(z)
    cdf = norm.cdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Diversity penalty: penalize candidates in well-explored (low sigma) regions.
    # The idea is that low sigma indicates the GP is confident, which often means
    # the area has been sampled densely. We want to discourage redundant evaluations.
    #
    # We define a "crowding" score based on inverse sigma. High crowding = low sigma.
    # To make it scale-invariant, we normalize sigma to [0, 1] using min-max normalization.
    sigma_min = np.min(sigma_safe)
    sigma_max = np.max(sigma_safe)
    
    if sigma_max - sigma_min > 1e-12:
        sigma_norm = (sigma_safe - sigma_min) / (sigma_max - sigma_min)
    else:
        sigma_norm = np.ones_like(sigma_safe)
    
    # Crowding is high when sigma_norm is low.
    # crowding = 1 - sigma_norm
    # We subtract a penalty proportional to crowding from the score.
    # The penalty should be scaled relative to the EI magnitude to be effective.
    ei_max = np.max(ei) if np.max(ei) > 1e-12 else 1.0
    ei_scale = ei_max
    
    # Penalty weight: how strongly to penalize crowded regions.
    # A weight of 0.5 means a fully crowded region (sigma_norm=0) loses half the max EI.
    penalty_weight = 0.5
    crowding_penalty = penalty_weight * (1.0 - sigma_norm) * ei_scale
    
    # Combined score: EI - crowding_penalty
    score = ei - crowding_penalty
    
    # Ensure scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Cost-aware: divide by cost (higher cost = lower priority)
    score = score / cost_safe
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
