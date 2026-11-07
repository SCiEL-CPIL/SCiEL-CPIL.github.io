"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Optimistic Lower Confidence Bound (LCB) for minimization.
    # We want to minimize f(x).
    # The optimistic lower bound on f(x) is mu - beta * sigma.
    # We select the candidate that minimizes this bound.
    # Since the evaluator picks the candidate with the HIGHEST score,
    # we return the negative of the LCB.
    # score = -(mu - beta * sigma) = -mu + beta * sigma
    
    # Beta controls the exploration-exploitation trade-off.
    # A higher beta favors exploration (high sigma).
    # A lower beta favors exploitation (low mu).
    # For multimodal functions like Ackley in a short horizon, 
    # a moderate-to-high beta helps escape local minima.
    beta = 2.5
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-9)
    
    scores = -mu + beta * sigma
    
    # Ensure finiteness
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
