import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 315.0

def acquisition(mu, sigma, f_best):
    if time.time() > _DEADLINE:
        return -mu.copy()
        
    sigma = np.maximum(sigma, 1e-8)
    
    # 1. EI
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # 2. Quantile UCB
    alpha = 1e-4
    k = -norm.ppf(alpha) # ~3.719
    ucb_q = -mu + k * sigma
    
    # 3. Normalize
    ei_max = np.max(ei)
    ei_norm = ei / ei_max if ei_max > 1e-12 else np.zeros_like(ei)
    
    ucb_min = np.min(ucb_q)
    ucb_max = np.max(ucb_q)
    ucb_range = ucb_max - ucb_min
    ucb_norm = (ucb_q - ucb_min) / ucb_range if ucb_range > 1e-12 else np.zeros_like(ucb_q)
    
    # 4. Combine
    # The aggressive k might make UCB dominate. 
    # Let's weight them.
    # If k is large, ucb_norm will be driven by sigma.
    # EI is driven by improvement.
    # Let's try 0.5/0.5 first, or maybe favor EI slightly more if k is very high?
    # Actually, if k is very high, the UCB term is very noisy.
    # Let's stick to 0.5/0.5 as in Insp 1, but with the new k.
    
    score = 0.5 * ei_norm + 0.5 * ucb_norm
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
