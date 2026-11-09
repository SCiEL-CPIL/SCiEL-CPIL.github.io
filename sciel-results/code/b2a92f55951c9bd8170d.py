import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Expected Improvement for minimization
    # EI(x) = E[max(0, f_best - f(x))]
    # where f(x) ~ N(mu, sigma^2)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # z = (f_best - mu) / sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return ei
