import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Thompson Sampling proxy: maximize (f_best - mu) / (sigma + epsilon)
    # This encourages points with high potential relative to their uncertainty.
    # Also add a small LCB component to ensure exploration.
    
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 300s limit
    
    epsilon = 1e-8
    improvement = f_best - mu
    normalized_improvement = improvement / (sigma + epsilon)
    
    # Combine with LCB for balance
    beta = 1.0
    lcb_score = -mu + beta * sigma
    
    # Weighted combination: 70% normalized improvement, 30% LCB
    # Normalize both to have similar scales
    ni_mean = np.mean(normalized_improvement)
    ni_std = np.std(normalized_improvement) + 1e-8
    ni_norm = (normalized_improvement - ni_mean) / ni_std
    
    lcb_mean = np.mean(lcb_score)
    lcb_std = np.std(lcb_score) + 1e-8
    lcb_norm = (lcb_score - lcb_mean) / lcb_std
    
    scores = 0.7 * ni_norm + 0.3 * lcb_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return scores
    
    return scores
