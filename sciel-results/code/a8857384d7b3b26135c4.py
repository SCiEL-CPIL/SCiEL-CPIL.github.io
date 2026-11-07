import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Calculate Lower Confidence Bound (LCB) score for minimization
    # We want to maximize this score, so we negate the LCB value (mu - beta*sigma)
    # Score = - (mu - beta*sigma) = -mu + beta*sigma
    beta = 2.0
    lcb_score = -mu + beta * sigma
    
    # Normalize scores to [0, 1] to balance scales
    # Add small epsilon to avoid division by zero
    eps = 1e-8
    
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    if ei_max > ei_min:
        ei_norm = (ei - ei_min) / (ei_max - ei_min + eps)
    else:
        ei_norm = np.zeros_like(ei)
        
    lcb_min = np.min(lcb_score)
    lcb_max = np.max(lcb_score)
    if lcb_max > lcb_min:
        lcb_norm = (lcb_score - lcb_min) / (lcb_max - lcb_min + eps)
    else:
        lcb_norm = np.zeros_like(lcb_score)
        
    # Blend EI and LCB scores
    # Weight: 0.7 for EI (exploitation-focused), 0.3 for LCB (exploration-focused)
    w_ei = 0.7
    w_lcb = 0.3
    
    scores = w_ei * ei_norm + w_lcb * lcb_norm
    
    # Deadline guard
    if time.time() - start_time > 1.0:
        # Fallback to simple exploitation
        scores = -mu
        
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
