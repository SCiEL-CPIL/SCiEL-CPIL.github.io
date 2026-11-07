"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # UCB (Upper Confidence Bound) acquisition for minimization.
    # Higher score means more promising to evaluate next.
    # We want to minimize, so we prefer low mu and high sigma.
    # UCB = mu - beta * sigma (lower is better for the objective, but we return higher score = more promising)
    # So score = -mu + beta * sigma, or equivalently we can think of it as:
    # We want to maximize the chance of finding a better (lower) value.
    # Using UCB: score = -mu + beta * sigma
    
    # Adaptive beta: start with high exploration, but keep it fixed for simplicity
    # Given the small budget (30 evaluations), high exploration is beneficial
    beta = 4.0
    
    # Clip sigma to avoid numerical issues
    sigma_clipped = np.clip(sigma, 1e-10, None)
    
    # UCB score for minimization: we want high scores for points likely to be low
    # score = -mu + beta * sigma
    score = -mu + beta * sigma_clipped
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
