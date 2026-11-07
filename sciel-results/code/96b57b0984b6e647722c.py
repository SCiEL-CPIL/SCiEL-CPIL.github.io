"""Rewritten program using UCB (Lower Confidence Bound) acquisition function."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Acquisition function based on the Lower Confidence Bound (LCB) for minimization.
    Score = -mu + k * sigma.
    Higher scores indicate points that are likely to have low function values
    (exploiting low mu) or high uncertainty (exploring high sigma).
    """
    # Fixed exploration parameter
    k = 1.0
    
    # Calculate LCB score for maximization
    # We want to minimize f(x), so we look for low predicted values.
    # The lower bound of the prediction is mu - k*sigma.
    # To select the best candidate by maximizing the acquisition score,
    # we use the negative of this lower bound: -(mu - k*sigma) = -mu + k*sigma.
    scores = -mu + k * sigma
    
    # Ensure all scores are finite
    # Replace any non-finite values (NaN, Inf) with a very low score
    scores = np.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
    
    return scores
# EVOLVE-BLOCK-END
