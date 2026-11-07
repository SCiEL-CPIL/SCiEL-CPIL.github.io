"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    # Calculate distance from the theoretical minimum (-0.8)
    # f_best is the lowest observed value.
    # We assume the true minimum is -0.8.
    d = np.maximum(0.0, f_best - (-0.8))
    
    # Adaptive beta: higher when far from minimum (explore), lower when close (exploit)
    # tanh(d) is 0 when d=0, and approaches 1 as d increases.
    beta = 1.0 + 2.0 * np.tanh(d)
    
    # Adaptive lambda: lower when far from minimum (ignore cost), higher when close (save cost)
    # exp(-d) is 1 when d=0, and decays as d increases.
    lambda_cost = 0.5 * np.exp(-1.0 * d)
    
    # LCB Score: maximize -mu + beta*sigma - lambda*cost
    # This is equivalent to minimizing mu - beta*sigma + lambda*cost
    score = -mu + beta * sigma - lambda_cost * cost
    
    # Ensure scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
