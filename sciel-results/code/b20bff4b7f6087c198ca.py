import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Worst-Case Improvement (WCI) score for minimization
    # WCI = max(0, f_best - mu - k * sigma)
    # This is a deterministic, robust criterion that penalizes high uncertainty.
    # It focuses on regions where we are confident the value is low (exploit)
    # but avoids regions with high variance (unlike LCB which might explore them).
    # We want to maximize this score.
    
    # Exploration constant k. 
    # A smaller k makes it more exploitative. 
    # Since we are using WCI (minimax regret), a moderate k is appropriate.
    # Let's try k=1.0 as a starting point, which is standard for such robust bounds.
    k = 1.0
    
    # Calculate raw WCI score
    wci = f_best - mu - k * sigma
    
    # Clip to 0 to ensure we only consider improvements
    scores = np.maximum(0.0, wci)
    
    # Cost-aware normalization
    # Cost is in [1, 2] (1 + mean(x) where x in [0,1])
    # Dividing by cost prioritizes cheaper evaluations
    cost_safe = np.maximum(cost, 1e-9)
    scores = scores / cost_safe
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    if time.time() - start_time > 290:
        return np.zeros_like(mu)
        
    return scores
