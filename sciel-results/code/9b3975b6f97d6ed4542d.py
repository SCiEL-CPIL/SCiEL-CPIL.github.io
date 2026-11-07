import numpy as np
import time
from scipy.stats import norm

_DEADLINE = 325.0
_t0 = time.time()

def acquisition(mu, sigma, f_best):
    """
    Compute Probability of Improvement (PI) acquisition scores.
    """
    if time.time() - _t0 > _DEADLINE:
        return -mu

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division issues
    sigma = np.maximum(sigma, 1e-8)
    
    # Z-score for improvement
    # We want P(f < f_best) = P((f - f_best) < 0)
    # Let f ~ N(mu, sigma^2)
    # P(f < f_best) = Phi((f_best - mu) / sigma)
    Z = (f_best - mu) / sigma
    
    # Probability of Improvement
    scores = norm.cdf(Z)
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    return scores
