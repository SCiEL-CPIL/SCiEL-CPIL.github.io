import numpy as np

def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) for minimization
    # We want to minimize the function, so we pick points with low mu and high sigma.
    # Score = -mu + beta * sigma
    beta = 2.0
    return -mu + beta * sigma
