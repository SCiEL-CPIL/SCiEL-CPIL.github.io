import numpy as np

def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) for minimization
    # We minimize mu(x) - beta * sigma(x)
    # Maximizing the negative of this: -mu(x) + beta * sigma(x)
    beta = 1.0
    return -mu + beta * sigma
