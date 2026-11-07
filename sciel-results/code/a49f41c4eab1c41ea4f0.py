import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # UCB for minimization: minimize mu - beta * sigma
    # Score = -mu + beta * sigma
    # Using sqrt(sigma) is also common (UCB usually uses sqrt(variance) or sigma)
    # Let's use sigma directly for simplicity, or sqrt.
    # Standard UCB uses sqrt(variance) = sigma.
    beta = 1.0
    return -mu + beta * sigma
