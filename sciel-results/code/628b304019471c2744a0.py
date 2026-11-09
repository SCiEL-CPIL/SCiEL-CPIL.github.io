import numpy as np

def acquisition(mu, sigma, f_best):
    # Use UCB (Upper Confidence Bound) for minimization.
    # Score = mu - beta * sigma
    # Higher score is better (we want low mu, high sigma).
    # beta controls the exploration/exploitation trade-off.
    # A value of 2.0 is a standard choice for moderate exploration.
    beta = 2.0
    scores = mu - beta * sigma
    return scores
