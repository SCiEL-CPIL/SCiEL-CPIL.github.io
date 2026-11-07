import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Computes the Upper Confidence Bound (UCB) acquisition function for minimization.
    Higher scores indicate better candidates.
    
    Args:
    mu: Array of predicted means.
    sigma: Array of predicted standard deviations.
    f_best: The best function value observed so far.
    
    Returns:
    Array of acquisition scores.
    """
    # UCB for minimization: score = beta * sigma - mu
    # We choose beta to balance exploration and exploitation.
    # A value of 2.5 is a common starting point for stiff functions.
    beta = 2.5
    
    # Ensure sigma is non-negative (though it should be from GP)
    sigma = np.maximum(sigma, 1e-9)
    
    scores = beta * sigma - mu
    
    return scores
