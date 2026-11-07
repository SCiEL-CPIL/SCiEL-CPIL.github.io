import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Ensure sigma is positive and non-zero to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Inverted UCB (for minimization, maximize -mu + beta * sigma)
    # Dynamic beta: higher when uncertainty is high (early), lower when low (late)
    # Use the median sigma as a proxy for exploration state
    median_sigma = np.median(sigma_safe)
    max_sigma = np.max(sigma_safe)
    
    # If max_sigma is very small, we are very late.
    # Normalize max_sigma to [0, 1] range roughly. 
    # If max_sigma > 1.0, we are likely early/mid.
    # Let's map max_sigma to a beta range [1.5, 4.0]
    # A simple heuristic: beta = 1.5 + 2.5 * (max_sigma / (max_sigma + 1.0))
    # This gives beta ~ 1.5 when max_sigma is small, ~ 4.0 when max_sigma is large.
    beta = 1.5 + 2.5 * (max_sigma / (max_sigma + 1.0))
    
    ucb_score = -mu + beta * sigma_safe
    
    # 3. Normalize both scores to [0, 1]
    eps = 1e-9
    
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range < eps:
        ei_norm = np.zeros_like(ei)
    else:
        ei_norm = (ei - ei_min) / ei_range
        
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    ucb_range = ucb_max - ucb_min
    if ucb_range < eps:
        ucb_norm = np.zeros_like(ucb_score)
    else:
        ucb_norm = (ucb_score - ucb_min) / ucb_range
        
    # 4. Combine with dynamic weights
    # If uncertainty (max_sigma) is high, trust UCB more.
    # If uncertainty is low, trust EI more.
    # weight_ucb should be higher when max_sigma is high.
    # Let's say w_ucb ranges from 0.3 (late) to 0.7 (early)
    w_ucb = 0.3 + 0.4 * (max_sigma / (max_sigma + 1.0))
    w_ei = 1.0 - w_ucb
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm
    
    return scores
