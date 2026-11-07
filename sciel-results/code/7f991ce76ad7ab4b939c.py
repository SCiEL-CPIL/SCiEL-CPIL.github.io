"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) strategy for minimization.
    # We aim to minimize the lower bound of the function value: mu - beta * sigma.
    # The acquisition score is defined such that higher values are preferred.
    # Therefore, we return the negative of the LCB: -mu + beta * sigma.
    
    # Beta controls the trade-off between exploration (sigma) and exploitation (mu).
    # A value of 2.0 is a common choice corresponding to a 95% confidence interval.
    beta = 2.0
    
    # Ensure sigma is non-negative to avoid numerical issues
    sigma = np.clip(sigma, 0.0, None)
    
    # Calculate the acquisition scores
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite to satisfy the contract
    # Points with non-finite scores are assigned a very low score to be avoided
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
