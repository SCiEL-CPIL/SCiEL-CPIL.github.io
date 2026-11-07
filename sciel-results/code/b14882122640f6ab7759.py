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
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Reset logic for new seed/run
    if _last_f_best is not None and f_best > _last_f_best:
        _call_count = 0
    _last_f_best = f_best
    _call_count += 1
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # 1. Compute EI
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Compute UCB (for minimization: -mu + beta * sigma)
    # Weierstrass is multimodal. Strong exploration early is crucial.
    # Beta 3.0 is more aggressive than 2.5, favoring high uncertainty regions.
    beta = 3.0
    ucb = -mu + beta * sigma
    
    # 3. Normalize to zero mean, unit variance (standardization)
    # This is more robust than min-max normalization which is sensitive to outliers.
    ei_mean = np.mean(ei)
    ei_std = np.std(ei)
    if ei_std < 1e-9:
        ei_norm = np.zeros_like(ei)
    else:
        ei_norm = (ei - ei_mean) / ei_std
        
    ucb_mean = np.mean(ucb)
    ucb_std = np.std(ucb)
    if ucb_std < 1e-9:
        ucb_norm = np.zeros_like(ucb)
    else:
        ucb_norm = (ucb - ucb_mean) / ucb_std
        
    # 4. Adaptive Weight
    # Structural change: Use a sharper transition to quickly switch from exploration to exploitation.
    # Early calls (0-5) should be heavily UCB to cover the space.
    # Late calls (15+) should be heavily EI to refine the best found.
    # The sigmoid here controls the blend.
    x = _call_count - 1 
    # Center the transition around step 10
    mid = 10
    # Steeper slope (k) for a more distinct phase change
    k = 0.8
    w = 1.0 / (1.0 + np.exp(-k * (x - mid)))
    
    # 5. Combine
    score = w * ei_norm + (1.0 - w) * ucb_norm
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return score
