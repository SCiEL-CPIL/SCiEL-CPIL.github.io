import numpy as np
import time
from scipy.stats import norm

_start_time = None
_call_count = 0

def acquisition(mu, sigma, f_best):
    global _start_time, _call_count
    if _start_time is None:
        _start_time = time.time()
    
    _call_count += 1
    
    # Deadline guard
    if time.time() - _start_time > 325:
        # Fallback to EI if time is tight
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.where(np.isfinite(ei), ei, 0.0)
        return ei
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Pure Decaying UCB Strategy
    # 30 total evaluations, 5 initial, so 25 optimization steps
    # Beta decays linearly from 3.0 (high exploration) to 0.5 (high exploitation)
    # This encourages covering the multimodal landscape early, then refining.
    progress = min(_call_count / 25.0, 1.0)
    beta = 3.0 - 2.5 * progress
    
    # For minimization, we want to maximize: -mu + beta * sigma
    # High sigma (uncertainty) and low mu (good prediction) yield high score
    ucb_score = -mu + beta * sigma_safe
    
    # Ensure finite scores
    ucb_score = np.where(np.isfinite(ucb_score), ucb_score, -np.inf)
    
    return ucb_score
