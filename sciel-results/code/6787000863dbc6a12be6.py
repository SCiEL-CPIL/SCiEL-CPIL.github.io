"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    # Use Lower Confidence Bound (LCB) for minimization, scaled by cost.
    # This is more exploratory than Expected Improvement (EI).
    # LCB(x) = mu(x) - beta * sigma(x)
    # We want to maximize the improvement: f_best - LCB(x) = f_best - mu(x) + beta * sigma(x)
    # Score = (f_best - mu + beta * sigma) / cost
    
    beta = 1.5
    scores = (f_best - mu + beta * sigma) / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
