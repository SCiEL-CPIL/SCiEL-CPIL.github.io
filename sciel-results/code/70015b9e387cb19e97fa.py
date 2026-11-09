import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Expected Improvement with exploration parameter xi
    xi = 0.01
    delta = f_best - xi - mu
    sigma_safe = np.maximum(sigma, 1e-9)
    z = delta / sigma_safe
    score = delta * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return score
