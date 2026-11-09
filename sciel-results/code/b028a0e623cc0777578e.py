import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # UCB with adaptive or fixed beta
    # Fixed beta 0.2 seems reasonable for smooth functions
    beta = 0.2
    scores = -mu + beta * sigma
    return scores
