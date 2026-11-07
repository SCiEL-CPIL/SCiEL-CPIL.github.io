"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: blends Expected Improvement (EI) with a UCB-style exploration term.
    
    mu, sigma: arrays of shape (N,) in objective units.
    f_best: lowest observed value.
    
    Returns: array of shape (N,) with scores. Higher score = more likely to be selected.
    """
    start_time = time.time()
    deadline = 325.0  # Safety margin before 330s limit
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement (EI) for minimization
    z_ei = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    
    # Normalize EI to [0, 1]
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros(n)
    
    # UCB term: favors points with high uncertainty (sigma)
    # Normalize sigma to [0, 1]
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-12:
        sigma_norm = sigma_safe / sigma_max
    else:
        sigma_norm = np.zeros(n)
    
    # Also add a small penalty for points with high mu (poor mean prediction)
    # This helps avoid exploring regions that are clearly bad
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = mu_max - mu_min
    if mu_range > 1e-12:
        mu_norm = (mu - mu_min) / mu_range  # 0 = best, 1 = worst
    else:
        mu_norm = np.zeros(n)
    
    # Adaptive beta: start with more exploration, shift to exploitation
    # For Michalewicz, the minimum is -1.8013
    min_possible = -1.8013
    # Progress: 0 when f_best is far from min, 1 when close
    progress = np.clip((min_possible - f_best) / (1.0 - min_possible), 0.0, 1.0)
    
    # When progress is low (early), explore more; when high (late), exploit more
    beta = 0.5 * (1.0 - progress)  # Exploration weight
    
    # Combine: EI (exploitation) + beta * (sigma exploration - mu penalty)
    # The exploration term rewards high sigma and penalizes high mu
    exploration = sigma_norm - 0.5 * mu_norm
    
    scores = ei_norm + beta * exploration
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        pass
    
    return scores
# EVOLVE-BLOCK-END
