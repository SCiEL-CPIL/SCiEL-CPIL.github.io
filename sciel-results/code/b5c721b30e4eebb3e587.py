"""Rewritten program using Lower Confidence Bound (LCB) acquisition function."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Computes the Lower Confidence Bound (LCB) acquisition scores.
    
    For minimization, we seek to minimize the lower bound of the function value:
    LCB(x) = mu(x) - beta * sigma(x)
    
    Since the evaluator selects the candidate with the HIGHEST score, we return:
    Score(x) = -LCB(x) = -mu(x) + beta * sigma(x)
    
    Parameters:
    mu (np.ndarray): Mean of the GP posterior, shape (2000,)
    sigma (np.ndarray): Standard deviation of the GP posterior, shape (2000,)
    f_best (float): The best observed function value so far. (Not used in LCB)
    
    Returns:
    np.ndarray: Scores for the 2000 candidates, shape (2000,)
    """
    # Beta controls the exploration-exploitation trade-off.
    # A higher beta encourages exploration.
    # In 1D with a local kernel (length scale 1.0 in original units),
    # we need sufficient exploration to find the global minimum at 0.
    beta = 2.0
    
    # Calculate LCB scores: -mu + beta * sigma
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite to prevent invalid results
    # If sigma is inf or mu is inf, handle gracefully
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
