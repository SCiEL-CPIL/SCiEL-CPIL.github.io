import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    sigma_safe = np.maximum(sigma, 1e-9)
    improvement = np.maximum(f_best - mu, 0.0)
    z = improvement / sigma_safe
    scores = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
