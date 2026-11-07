import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function for minimization.
    Returns scores where higher values indicate more promising candidates.
    """
    beta = 2.0
    # Minimize: mu - beta * sigma
    # Maximize: -mu + beta * sigma
    scores = -mu + beta * sigma
    return scores
