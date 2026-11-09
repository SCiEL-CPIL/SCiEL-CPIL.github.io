import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Use Lower Confidence Bound (LCB) for minimization
    # This encourages exploration in high-dimensional spaces like Hartmann 6D
    # where standard EI might be too greedy.
    beta = 2.0
    return -mu + beta * sigma
