import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() - _START_TIME > 320:
        return np.zeros_like(mu)
    
    # Pareto-frontier guided exploration
    # 1. Identify the exploitation anchor: candidate with the lowest predicted mean
    mu_min = np.min(mu)
    idx_anchor = np.argmin(mu)
    sigma_anchor = sigma[idx_anchor]
    
    # 2. Compute distance in prediction space (mu, sigma) from the anchor
    d_mu = mu - mu_min
    d_sigma = sigma - sigma_anchor
    distances = np.sqrt(d_mu**2 + d_sigma**2)
    
    # 3. Normalize distances to [0, 1]
    max_dist = np.max(distances)
    if max_dist > 1e-8:
        norm_distances = distances / max_dist
    else:
        norm_distances = np.zeros_like(distances)
    
    # 4. Score is the product of standard deviation and normalized distance
    scores = sigma * norm_distances
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
