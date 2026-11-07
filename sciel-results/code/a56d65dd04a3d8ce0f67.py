import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # 1. Compute EI
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Handle numerical stability
    ei = np.where(sigma < 1e-9, np.where(mu < f_best, f_best - mu, 0.0), ei)
    ei = np.maximum(ei, 0.0) # EI should be non-negative
    
    # 2. Find Pareto Front
    # A point is non-dominated if no other point has higher EI and lower cost.
    n = len(mu)
    is_front = np.ones(n, dtype=bool)
    
    # Efficient way: Sort by EI descending, then scan for min cost
    # If a point has a lower cost than all previous points (with higher EI), it is on the front.
    order = np.argsort(-ei) # Descending EI
    ei_sorted = ei[order]
    cost_sorted = cost[order]
    
    min_cost_so_far = np.inf
    front_mask_sorted = np.zeros(n, dtype=bool)
    for i in range(n):
        if cost_sorted[i] < min_cost_so_far:
            front_mask_sorted[i] = True
            min_cost_so_far = cost_sorted[i]
            
    # Map back to original indices
    front_mask = np.zeros(n, dtype=bool)
    front_mask[order] = front_mask_sorted
    
    # 3. Score the Front
    # Normalize EI and Cost
    ei_min, ei_max = np.min(ei), np.max(ei)
    cost_min, cost_max = np.min(cost), np.max(cost)
    
    if ei_max > ei_min:
        ei_norm = (ei - ei_min) / (ei_max - ei_min)
    else:
        ei_norm = np.zeros_like(ei)
        
    if cost_max > cost_min:
        cost_norm = (cost - cost_min) / (cost_max - cost_min)
    else:
        cost_norm = np.zeros_like(cost)
        
    # Ideal point: EI=1, Cost=0
    # Distance to ideal
    dist = np.sqrt((1 - ei_norm)**2 + cost_norm**2)
    
    scores = np.zeros_like(ei)
    # Assign scores to front points
    scores[front_mask] = 1.0 / (1.0 + dist[front_mask])
    
    # Optional: Add a small bonus to the absolute best EI point if it's on the front?
    # The distance metric should handle it.
    
    # Deadline check
    elapsed = time.time() - start_time
    if elapsed > 280: # Safety margin
         # Return something valid, e.g., the EI ratio
         return ei / (cost + 1e-9)
         
    return scores
