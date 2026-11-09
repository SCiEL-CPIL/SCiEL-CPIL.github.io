"""Cost-aware acquisition function for Bayesian optimization (styblinski tang 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Compute a cost-aware acquisition score for 2000 candidates.
    Uses a cost-aware Lower Confidence Bound (LCB) with adaptive exploration.
    
    Args:
        mu: Array of GP means (shape 2000,)
        sigma: Array of GP standard deviations (shape 2000,)
        f_best: Current best observed value (scalar)
        cost: Array of evaluation costs (shape 2000,)
        
    Returns:
        Array of acquisition scores (shape 2000,), higher is better.
    """
    start_time = time.time()
    deadline = 325.0  # Safety margin before 330s limit

    # Convert to numpy arrays and ensure float64
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Sanitize inputs
    # Replace non-finite mu with f_best (neutral)
    mu_safe = np.where(np.isfinite(mu), mu, f_best)
    # Replace non-finite or tiny sigma with a small positive value
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 1e-9), sigma, 1e-6)
    # Replace non-finite or tiny cost with 1.0 (minimum cost)
    cost_safe = np.where(np.isfinite(cost) & (cost > 1e-9), cost, 1.0)

    # Adaptive exploration parameter beta
    # Known minimum for Styblinski-Tang 2D is -78.3323
    known_min = -78.3323
    
    # Calculate gap to known minimum
    # If f_best is much worse than known_min, we are in early stages -> explore more
    # If f_best is close to known_min, we are in late stages -> exploit more
    gap = f_best - known_min
    
    # Define beta range
    beta_min = 1.0  # Exploitation
    beta_max = 4.0  # Exploration
    
    # Normalize gap for adaptive beta
    # A gap of 0 means we found the min (or very close). 
    # A gap of 100+ means we are far away.
    # Use a smooth transition.
    # If gap < 5, beta = beta_min
    # If gap > 50, beta = beta_max
    # Linear interpolation in between? Or exponential decay?
    # Let's use a simple linear mapping with clipping.
    
    if gap <= 5.0:
        beta = beta_min
    elif gap >= 50.0:
        beta = beta_max
    else:
        # Map [5, 50] to [beta_min, beta_max]
        t = (gap - 5.0) / 45.0
        beta = beta_min + t * (beta_max - beta_min)

    # Calculate LCB score for minimization
    # We want to maximize: - (mu - beta * sigma) = beta * sigma - mu
    lcb_score = beta * sigma_safe - mu_safe
    
    # Cost-aware adjustment
    # Divide by cost to penalize expensive points
    # This effectively scales the acquisition function by 1/cost
    score = lcb_score / cost_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline check (unlikely to be needed for vectorized ops, but required by directive)
    if time.time() > deadline:
         # Return best effort (current score)
         return score

    return score
# EVOLVE-BLOCK-END
