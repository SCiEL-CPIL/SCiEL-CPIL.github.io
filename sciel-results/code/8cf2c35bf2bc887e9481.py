import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIMEOUT = 320.0 # 10s margin

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() - _start_time > _TIMEOUT:
        # Return a safe fallback, e.g., EI
        sigma_safe = sigma + 1e-9
        z = (f_best - mu) / sigma_safe
        return (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    sigma_safe = sigma + 1e-9
    
    # EI
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB with beta=1.0 (moderate exploration)
    # Score to maximize: -mu + beta*sigma
    ucb = -mu + 1.0 * sigma_safe
    
    # Normalize
    def norm_scores(s):
        m = np.mean(s)
        std = np.std(s)
        if std < 1e-9:
            return s - m
        return (s - m) / std
        
    z_ei = norm_scores(ei)
    z_ucb = norm_scores(ucb)
    
    # Weighted sum: Favor EI
    final_scores = 0.7 * z_ei + 0.3 * z_ucb
    
    return final_scores
