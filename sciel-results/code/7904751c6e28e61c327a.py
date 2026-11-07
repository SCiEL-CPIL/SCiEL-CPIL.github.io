import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Structural Change: Use UCB (Upper Confidence Bound) instead of EI
    # UCB encourages exploration of regions with high uncertainty (sigma)
    # while still considering the expected value (mu).
    # This can help escape local minima in functions like Powell 4D.
    
    # Fixed exploration-exploitation balance parameter
    beta = 2.0
    
    # Calculate UCB
    ucb = mu + beta * sigma
    
    # Cost-aware normalization
    # We divide by the cost to prefer cheaper evaluations that offer similar UCB.
    # This respects the budget constraint.
    safe_cost = np.maximum(cost, 1e-9)
    scores = ucb / safe_cost
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Deadline guard
    # If computation takes unexpectedly long, return a safe fallback
    if time.time() - start_time > 0.5:
        # Return sigma (pure exploration) as a safe, finite fallback
        return sigma
        
    return scores
