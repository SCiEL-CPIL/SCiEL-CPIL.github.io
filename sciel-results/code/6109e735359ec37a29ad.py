import numpy as np

def acquisition(mu, sigma, f_best):
    # GP-UCB for minimization: score = mu - beta * sigma
    # beta = 1.0 based on measured improvement in 6D space
    beta = 1.0
    scores = mu - beta * sigma
    return scores
