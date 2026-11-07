import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Ensure sigma is positive to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    
    # 1. Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # 2. Lower Confidence Bound (LCB)
    # For minimization, we want low mu and high sigma.
    # Score should be high when (mu - beta * sigma) is low.
    # So lcb_score = -mu + beta * sigma
    beta = 1.0
    lcb_score = -mu + beta * sigma
    
    # 3. Probability of Improvement (POI)
    poi = norm.cdf(z)
    
    # Normalize EI to [0, 1]
    # EI is non-negative. If max is 0, all are 0.
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range > 1e-9:
        ei_norm = (ei - ei_min) / ei_range
    else:
        ei_norm = np.zeros_like(ei)
        
    # Normalize LCB score to [0, 1]
    lcb_min = np.min(lcb_score)
    lcb_max = np.max(lcb_score)
    lcb_range = lcb_max - lcb_min
    if lcb_range > 1e-9:
        lcb_norm = (lcb_score - lcb_min) / lcb_range
    else:
        lcb_norm = np.zeros_like(lcb_score)
        
    # POI is already in [0, 1]
    
    # Combine scores
    # Weights from the high-scoring inspiration: 0.5 EI, 0.3 LCB, 0.2 POI
    # This balances exploitation (EI), exploration (LCB), and quick wins (POI)
    score = 0.5 * ei_norm + 0.3 * lcb_norm + 0.2 * poi
    
    # Ensure scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
