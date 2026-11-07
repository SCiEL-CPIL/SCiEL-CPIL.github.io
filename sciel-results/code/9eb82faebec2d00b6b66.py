import numpy as np
from scipy.stats import norm
import time

_call_count = 0
_last_f_best = None
_start_time = time.time()

def acquisition(mu, sigma, f_best):
    global _call_count, _last_f_best, _start_time
    
    # Deadline guard
    if time.time() - _start_time > 320:
        # Fallback to EI
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei

    # Reset logic for new seed/run
    if _last_f_best is not None and f_best > _last_f_best:
        _call_count = 0
    _last_f_best = f_best
    _call_count += 1
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Structural Change: Use Log-Expected Improvement
    # This compresses the scale of EI, potentially leading to more stable
    # selection of candidates with moderate but reliable improvements
    # over those with high uncertainty but low probability of large gain.
    # We add a small epsilon to avoid log(0).
    log_ei = np.log(ei + 1e-6)
    
    # Ensure finite
    score = np.nan_to_num(log_ei, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
