"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    # Use Upper Confidence Bound (UCB) for minimization.
    # We want to find points that likely have low function values.
    # The lower bound of the confidence interval is mu - beta * sigma.
    # We want to minimize this lower bound, so we maximize -(mu - beta * sigma) = -mu + beta * sigma.
    # A beta of 2.5 provides a good balance of exploration and exploitation for this short horizon.
    beta = 2.5
    scores = -mu + beta * sigma
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
