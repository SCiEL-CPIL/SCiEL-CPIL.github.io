import numpy as np

def acquisition(mu, sigma, f_best):
    # UCB acquisition function for minimization
    # We want to maximize this score.
    # -mu encourages exploitation (low predicted mean)
    # + beta * sigma encourages exploration (high uncertainty)
    beta = 1.0
    return -mu + beta * sigma
