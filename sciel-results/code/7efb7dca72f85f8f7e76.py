"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware UCB (Upper Confidence Bound) for minimization.
    # We maximize (f_best - mu + beta * sigma) / cost.
    # This encourages exploration in areas with high uncertainty (sigma),
    # which is crucial for multimodal functions like Rastrigin where the GP
    # might smooth out local minima, leading to high mu in unexplored regions.
    # EI can be too exploitative in this regime.
    
    beta = 1.5
    
    # Calculate UCB-like score
    # f_best - mu is the expected improvement if we exploit.
    # beta * sigma is the exploration bonus.
    score = (f_best - mu + beta * sigma) / cost
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
