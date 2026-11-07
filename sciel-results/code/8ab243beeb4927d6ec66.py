import numpy as np
from scipy.stats import norm
import time

_t0 = time.time()

def acquisition(mu, sigma, f_best):
    # Deadline guard: if we're running long, fall back to greedy
    if time.time() - _t0 > 290.0:
        return -np.asarray(mu, dtype=float).copy()
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    eps = 1e-9
    s = sigma + eps
    z = (f_best - mu) / s
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # Exploration bonus: reward uncertainty where improvement is plausible
    bonus = 0.3 * sigma * norm.sf(z)
    scores = ei + bonus
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
