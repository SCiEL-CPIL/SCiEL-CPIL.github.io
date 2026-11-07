import numpy as np

def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound for minimization
    # Score = -mu + kappa * sigma
    # Higher score is better.
    kappa = 1.5
    return -mu + kappa * sigma
