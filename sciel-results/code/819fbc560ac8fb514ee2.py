import numpy as np
import time

# Global time tracking for safety
_start_time = time.time()
_time_limit = 300.0 # Seconds

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we are dangerously close to the limit, return a safe fallback
    if time.time() - _start_time > _time_limit - 5.0:
        return np.zeros_like(mu)
    
    # Safe sigma to avoid division by zero
    safe_sigma = np.maximum(sigma, 1e-9)
    safe_cost = np.maximum(cost, 1e-9)
    
    # Cost-aware UCB acquisition function
    # For minimization, we want to find points with low potential values.
    # UCB for minimization: potential_value = mu - beta * sigma
    # (We subtract beta*sigma because lower values are better, and uncertainty
    #  could mean the true value is even lower than mu)
    
    # The score should be higher for more promising points.
    # A point is promising if it could be much better than f_best.
    # Let's define the "improvement potential" as f_best - (mu - beta*sigma)
    # = (f_best - mu) + beta*sigma
    
    # But we also need to account for cost.
    # Cost-aware score: improvement_potential / cost
    
    # Let's use beta = 2.0 as suggested
    beta = 2.0
    
    # Improvement potential: how much better this point could be than f_best
    # Lower bound on f(x) is approximately mu - beta*sigma
    # So potential improvement = f_best - (mu - beta*sigma) = (f_best - mu) + beta*sigma
    
    improvement_potential = (f_best - mu) + beta * safe_sigma
    
    # We want to maximize this, but also penalize high cost
    # Score = improvement_potential / cost
    
    score = improvement_potential / safe_cost
    
    # Handle any numerical issues
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
