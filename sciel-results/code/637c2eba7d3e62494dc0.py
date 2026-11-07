import numpy as np

def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound for minimization
    # We want to maximize (beta * sigma - mu)
    # This is equivalent to minimizing (mu - beta * sigma)
    beta = 1.0
    scores = beta * sigma - mu
    return scores
