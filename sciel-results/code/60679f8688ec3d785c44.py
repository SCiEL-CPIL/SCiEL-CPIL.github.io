import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Sanitize inputs
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.maximum(sigma, 1e-6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=1e-6)
    cost = np.maximum(cost, 1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1.0)
    
    # Thompson Sampling: Sample from the posterior
    # y ~ N(mu, sigma^2)
    # Use a fixed random seed? No, randomness is key for TS.
    # However, for reproducibility in debugging, we might consider it, 
    # but the task implies stochastic search is allowed/encouraged.
    y = mu + sigma * np.random.normal(size=mu.shape)
    
    # Calculate Improvement
    # We want to minimize, so improvement is f_best - y if y < f_best
    improvement = np.maximum(0.0, f_best - y)
    
    # Cost-aware score
    # Higher is better. 
    # We divide by cost to prefer cheaper evaluations.
    scores = improvement / cost
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    
    # Safety margin check (though unlikely to trigger for 2000 points)
    if time.time() - start_time > 290.0:
        # Fallback to a simpler metric if this takes too long (shouldn't happen)
        return scores
        
    return scores
