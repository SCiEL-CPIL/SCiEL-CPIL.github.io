"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Cost-aware LCB strategy
    # Standard LCB for minimization: mu - beta * sigma
    # We want to maximize the score, so we minimize (mu - beta * sigma) / cost
    # This is equivalent to maximizing (beta * sigma - mu) / cost
    # Note: f_best is not explicitly used in standard LCB, but we can use it 
    # to adapt beta if needed. However, the insight says pure EI/cost or UCB 
    # variants are strong. Let's stick to a robust LCB form.
    
    # A common adaptive beta is sqrt(log(n)) or similar, but we don't have n.
    # However, we have 2000 candidates. 
    # Let's try a fixed beta that balances exploration well for 3D Hartmann.
    # Typical beta values for LCB are 1.0 to 5.0.
    # Since we are dividing by cost, we effectively penalize expensive points more.
    
    # Alternative: Use the "Knowledge Gradient" or "Expected Improvement" is already tried.
    # The prompt mentions "Adaptive UCB" improved a different island.
    # Let's implement a cost-aware LCB.
    
    # We need to return scores where higher is better.
    # LCB value = mu - beta * sigma
    # We want to find the point with the lowest LCB value, adjusted for cost.
    # Score = -LCB / cost = (beta * sigma - mu) / cost
    
    beta = 2.5  # Exploration parameter
    
    lcb = mu - beta * sigma_safe
    scores = -lcb / cost
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 290.0:
         # Fallback if somehow slow (unlikely for vectorized ops)
         pass
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
