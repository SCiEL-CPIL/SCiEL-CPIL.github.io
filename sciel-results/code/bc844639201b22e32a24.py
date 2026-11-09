import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound (UCB) for minimization
    # We want to maximize the score, which corresponds to minimizing the function.
    # Score = -mu + beta * sigma
    # A higher score means a lower expected value or higher uncertainty.
    beta = 2.0
    return -mu + beta * sigma
