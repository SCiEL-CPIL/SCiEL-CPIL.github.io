import numpy as np

def acquisition(mu, sigma, f_best):
    # UCB with higher exploration to handle sparse sampling relative to length scale
    beta = 1.0
    scores = -mu + beta * sigma
    return scores
