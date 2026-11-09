"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Lower Confidence Bound (LCB) for minimization.
    # We maximize the expected improvement in the lower bound of the confidence interval,
    # normalized by the evaluation cost.
    # This approach is often more robust in high dimensions than EI as it
    # balances exploration (via sigma) and exploitation (via mu) linearly.
    
    beta = 2.0  # Exploration parameter
    
    # Calculate the lower bound of the confidence interval for the objective value
    # Since we are minimizing, a "good" point has a low value.
    # We want to maximize the potential improvement: f_best - (mu + beta * sigma)
    lcb_value = f_best - (mu + beta * sigma)
    
    # Normalize by cost to prefer cheaper evaluations
    # Add a small epsilon to cost just in case, though cost >= 1 here
    scores = lcb_value / (cost + 1e-9)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
