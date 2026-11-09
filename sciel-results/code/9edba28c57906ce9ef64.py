import numpy as np

def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB)
    # We want to minimize the function, so we prefer low mu.
    # We also want to explore areas with high uncertainty (sigma).
    # The score is -mu - beta * sigma. Higher score is better.
    # A fixed beta of 3.0 is a common choice for balanced exploration/exploitation.
    beta = 3.0
    scores = -mu - beta * sigma
    return scores
