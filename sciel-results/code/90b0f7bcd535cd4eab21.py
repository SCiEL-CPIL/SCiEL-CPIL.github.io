import numpy as np
import time

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    
    # Pareto-frontier guided exploration:
    # 1. Find the candidate with the lowest predicted mean (exploitation anchor)
    # 2. Score = sigma * normalized_distance_from_anchor
    # This prioritizes high-uncertainty regions far from the current best prediction
    
    # Ensure sigma is non-negative and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Find the anchor: index with minimum mu (best exploitation point)
    anchor_idx = np.argmin(mu)
    anchor_mu = mu[anchor_idx]
    anchor_sigma = sigma_safe[anchor_idx]
    
    # Compute normalized distance from anchor in prediction space
    # Distance combines both mean and uncertainty differences
    mu_diff = np.abs(mu - anchor_mu)
    sigma_diff = np.abs(sigma_safe - anchor_sigma)
    
    # Normalize by the range to make scale-invariant
    mu_range = np.ptp(mu)
    sigma_range = np.ptp(sigma_safe)
    
    # Avoid division by zero
    mu_range = max(mu_range, 1e-9)
    sigma_range = max(sigma_range, 1e-9)
    
    mu_norm = mu_diff / mu_range
    sigma_norm = sigma_diff / sigma_range
    
    # Combined distance metric
    dist = np.sqrt(mu_norm**2 + sigma_norm**2)
    
    # Score: high uncertainty AND far from the current best exploitation point
    # This encourages exploring unexplored regions with high uncertainty
    scores = sigma_safe * dist
    
    # Add a small exploitation component to ensure we don't completely ignore
    # points that are likely to improve on f_best
    # Blend: 80% exploration (distance-weighted uncertainty), 20% exploitation
    exploit_score = -mu
    scores = 0.8 * scores + 0.2 * exploit_score
    
    # Deadline guard
    if time.time() - start_time > 1.0:
        # Fallback: Greedy (minimize mu)
        scores = -mu
        
    return scores
