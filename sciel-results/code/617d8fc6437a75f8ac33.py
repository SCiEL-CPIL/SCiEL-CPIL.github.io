import numpy as np
import time

def acquisition(mu, sigma, f_best):
    # Deadline guard
    start = time.time()
    limit = 1.0 # 1 second limit per call, plenty for N=2000
    
    # Entropy Maximization / Information Gain Acquisition
    # Score based on log of posterior variance to target high uncertainty regions
    # This helps map the multimodal landscape of Beale function more efficiently
    # than confidence bounds which can be biased by current mean estimates
    
    # Ensure sigma is non-negative for safety
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Use log variance (proportional to log sigma^2) as the score
    # log(sigma^2) = 2 * log(sigma)
    # Higher uncertainty -> higher score
    scores = 2.0 * np.log(sigma_safe)
    
    # If it takes too long (unlikely), we could fallback
    if time.time() - start > limit:
         # Fallback to UCB if log fails or takes too long
         beta = 5.0
         scores = -mu + beta * sigma_safe
         
    return scores
