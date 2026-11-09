import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 320.0  # Safety margin before 330s kill

def _time_remaining():
    return max(0.0, _TIME_LIMIT - (time.time() - _START_TIME))

def _sanitize(scores):
    """Ensure scores are valid: finite, correct shape."""
    if scores.shape != (2000,):
        scores = np.resize(scores, (2000,))
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    return scores

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if _time_remaining() < 5.0:
        # Return EI as fallback (fast path)
        sigma_safe = np.maximum(np.nan_to_num(sigma, nan=1e-9, posinf=1e6, neginf=1e-9), 1e-9)
        mu_safe = np.nan_to_num(mu, nan=f_best, posinf=f_best, neginf=f_best)
        z = (f_best - mu_safe) / sigma_safe
        ei = (f_best - mu_safe) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return _sanitize(ei)
    
    # Sanitize inputs
    mu_safe = np.nan_to_num(mu, nan=f_best, posinf=f_best, neginf=f_best)
    sigma_safe = np.maximum(np.nan_to_num(sigma, nan=1e-9, posinf=1e6, neginf=1e-9), 1e-9)
    f_best_safe = np.nan_to_num(f_best, nan=-1e6, posinf=1e6, neginf=-1e6)
    
    # Compute Expected Improvement for minimization
    z = (f_best_safe - mu_safe) / sigma_safe
    # Numerically stable EI
    ei = (f_best_safe - mu_safe) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute LCB (to be maximized) with adaptive beta
    # Use a fixed beta that balances exploration and exploitation
    # For 30 evaluations, beta around 1.5-2.0 works well
    beta = 1.8
    lcb_max = beta * sigma_safe - mu_safe
    
    # Robust normalization using rank-based approach to avoid sensitivity to outliers
    def _rank_normalize(arr):
        """Normalize using ranks to [0, 1] for robustness."""
        order = np.argsort(arr)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(arr), dtype=np.float64)
        if len(arr) > 1:
            ranks = ranks / (len(arr) - 1)
        else:
            ranks = np.ones_like(arr)
        return ranks
    
    ei_norm = _rank_normalize(ei)
    lcb_norm = _rank_normalize(lcb_max)
    
    # Hybrid score: blend EI and LCB
    # Weight towards EI for exploitation, LCB for exploration
    score = 0.6 * ei_norm + 0.4 * lcb_norm
    
    return _sanitize(score)
