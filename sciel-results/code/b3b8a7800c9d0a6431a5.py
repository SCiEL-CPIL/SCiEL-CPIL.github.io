"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    # Use Lower Confidence Bound (LCB) for minimization.
    # We want to minimize the function value.
    # The optimistic lower bound is mu - beta * sigma.
    # We select the candidate with the lowest lower bound.
    # Since the evaluator selects the candidate with the highest score,
    # we return the negative of the lower bound: -mu + beta * sigma.
    
    beta = 3.0
    
    # Ensure sigma is non-negative to avoid numerical issues
    sigma = np.maximum(sigma, 0.0)
    
    # Calculate the score
    scores = -mu + beta * sigma
    
    return scores
# EVOLVE-BLOCK-END
