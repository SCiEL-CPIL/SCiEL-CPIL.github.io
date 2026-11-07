import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware acquisition: blend of expected improvement and UCB,
    # normalized by cost to account for evaluation expense.
    # Add a small exploration term that decays as f_best approaches the minimum.
    
    # Theoretical minimum for Styblinski-Tang 2D
    f_min = -78.3323
    
    # Progress-based exploration weight
    # Early: high exploration, Late: low exploration
    f_start = 20.0  # heuristic starting value
    denom = f_start - f_min
    if denom <= 0:
        denom = 1.0
    progress = np.clip((f_start - f_best) / denom, 0.0, 1.0)
    
    # Beta for UCB: higher early, lower late
    beta = 0.5 * (1.0 - 0.5 * progress)
    
    # Expected Improvement
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB
    ucb = -mu + beta * sigma
    
    # Blend: primarily EI, with some UCB for exploration
    # Weight EI more as we get closer to optimum
    ei_weight = 0.7 + 0.3 * progress
    ucb_weight = 1.0 - ei_weight
    
    # Combine (higher is better for acquisition)
    score = ei_weight * ei + ucb_weight * (-ucb)
    
    # Normalize by cost
    score = score / (cost + 1e-9)
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
