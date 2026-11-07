"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Adaptive UCB acquisition for minimization
    # Score = -mu + beta * sigma
    # We want to maximize this score.
    # 
    # Hypothesis: Use a beta that is higher when f_best is large (early stage, high uncertainty/progress needed)
    # and lower when f_best is small (late stage, close to optimum).
    # 
    # f_best is the lowest observed value.
    # Ackley 2D min is 0.
    # Early on, f_best might be ~5-10. Late on, f_best ~0.
    #
    # beta = base + range * (1 - exp(-k * f_best))
    # If f_best is large, exp(-k*f_best) is small, term is ~1. beta is high.
    # If f_best is small, exp(-k*f_best) is ~1, term is ~0. beta is low.
    
    # Parameters:
    base_beta = 0.5
    max_extra_beta = 3.0
    k = 0.5
    
    # Ensure f_best is non-negative for the formula stability, though it should be.
    fb = max(0.0, float(f_best))
    
    # Calculate adaptive beta
    beta = base_beta + max_extra_beta * (1.0 - np.exp(-k * fb))
    
    # UCB score for minimization:
    # We want to find points with low mu (good mean) and high sigma (high uncertainty).
    # Minimize mu - beta * sigma  <=>  Maximize -mu + beta * sigma
    scores = -mu + beta * sigma
    
    return scores
# EVOLVE-BLOCK-END
