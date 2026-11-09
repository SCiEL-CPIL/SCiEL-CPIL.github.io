# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function for minimization.
    
    Args:
        mu: array of shape (2000,) with posterior means in objective units.
        sigma: array of shape (2000,) with posterior standard deviations in objective units.
        f_best: float, the lowest observed value so far (unused in standard UCB).
    
    Returns:
        array of shape (2000,) with scores; higher means more promising to evaluate next.
    """
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0)
    
    # UCB score for minimization: we want to find low values.
    # The lower bound of the function value is mu - beta * sigma.
    # We want to maximize the potential for a low value, so we maximize - (mu - beta * sigma).
    # Score = -mu + beta * sigma
    
    beta = 1.0
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
