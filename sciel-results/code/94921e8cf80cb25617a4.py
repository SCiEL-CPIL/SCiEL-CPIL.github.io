import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Expected improvement per unit square root of cost.
    # This reduces the penalty on expensive points compared to 1/cost,
    # allowing better exploration in regions with higher evaluation cost.
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    return ei / np.sqrt(cost + 1e-9)
