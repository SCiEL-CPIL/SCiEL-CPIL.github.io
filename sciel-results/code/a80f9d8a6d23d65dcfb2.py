import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Calculate Lower Confidence Bound (LCB) score for minimization
    # We want to maximize this score, so we negate the LCB value (mu - beta*sigma)
    # Score = - (mu - beta*sigma) = -mu + beta*sigma
    beta = 3.0
    lcb_score = -mu + beta * sigma
    
    # Normalize scores to [0, 1] to balance scales
    # Add small epsilon to avoid division by zero in range
    eps = 1e-9
    
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min + eps
    ei_norm = (ei - ei_min) / ei_range
    
    lcb_min = np.min(lcb_score)
    lcb_max = np.max(lcb_score)
    lcb_range = lcb_max - lcb_min + eps
    lcb_norm = (lcb_score - lcb_min) / lcb_range
    
    # Combine with weights (0.7 LCB, 0.3 EI)
    # Higher weight on LCB for exploration in multimodal landscape
    scores = 0.7 * lcb_norm + 0.3 * ei_norm
    
    return scores
