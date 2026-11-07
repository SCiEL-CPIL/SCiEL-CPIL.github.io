import numpy as np

def acquisition(mu, sigma, f_best):
    # UCB: -mu + beta * sigma
    # Higher score is better.
    # We are minimizing, so we want low mu and high sigma.
    # Score = -mu + 2.0 * sigma
    beta = 2.0
    scores = -mu + beta * sigma
    return scores
