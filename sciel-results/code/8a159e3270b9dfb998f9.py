import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Use Upper Confidence Bound (UCB) for minimization.
    # We want to minimize mu - beta * sigma.
    # So we maximize beta * sigma - mu.
    beta = 2.0
    score = beta * sigma - mu
    # Ensure finite
    score = np.where(np.isfinite(score), score, -np.inf)
    return score
