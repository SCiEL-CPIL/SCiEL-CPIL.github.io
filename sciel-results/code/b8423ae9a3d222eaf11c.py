import numpy as np

def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) for minimization
    # We want to maximize the score, which corresponds to minimizing the objective.
    # Score = -mu + beta * sigma
    beta = 2.0
    return -mu + beta * sigma
