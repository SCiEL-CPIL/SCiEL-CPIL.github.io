import numpy as np

def acquisition(mu, sigma, f_best):
    # Upper Confidence Bound for minimization
    # We want to maximize the potential for finding a lower value.
    # Score = beta * sigma - mu
    # Using a fixed beta.
    beta = 2.0
    scores = beta * sigma - mu
    return scores
