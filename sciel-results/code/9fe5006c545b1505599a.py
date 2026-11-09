import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle zero sigma to avoid issues, though it shouldn't happen often
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Thompson Sampling:
    # Sample a single function value from the posterior distribution N(mu, sigma^2)
    # For minimization, we want low values.
    # y = mu + sigma * z
    z = np.random.standard_normal(len(mu))
    y = mu + sigma_safe * z
    
    # Cost-aware selection:
    # We want to minimize y / cost
    # So we return a score where higher is better.
    # Score = - (y / cost)
    # The candidate with the lowest y/cost will have the highest score (closest to 0 or positive if y is negative enough).
    
    scores = - (y / cost_safe)
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Deadline guard
    if time.time() - start_time > 325:
        # Fallback to a simple deterministic metric if we run out of time (unlikely for this simple op)
        # But we must return finite scores.
        return np.zeros_like(mu)
        
    return scores
