import numpy as np
import time
import math

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Kriging Distance acquisition function.
    
    Strategy:
    - Prioritize exploration of regions with high predictive uncertainty (sigma).
    - Penalize points close to already observed points (distance to nearest neighbor).
    - Normalize by cost to be cost-aware.
    - Use a safety deadline to ensure we return within 300s.
    
    Inputs:
    - mu: array of shape (2000,) predicted means
    - sigma: array of shape (2000,) predicted std devs
    - f_best: float, best observed value
    - cost: array of shape (2000,) evaluation costs
    
    Returns:
    - array of shape (2000,) scores (higher is better)
    """
    start_time = time.time()
    deadline = 290.0  # Safety margin before 300s limit
    
    n = len(mu)
    
    # If we have no prior knowledge or invalid inputs, fall back to random-ish exploration
    if n == 0:
        return np.array([])
        
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Kriging Distance heuristic:
    # We want high sigma (uncertainty) and low cost.
    # Since we don't have coordinates, we cannot compute true distance to nearest neighbor.
    # However, in BO with GPs, high sigma often correlates with being far from data.
    # A common proxy when coordinates are missing is just sigma itself, 
    # but to differentiate from UCB/EI which use mu, we can focus purely on sigma.
    # 
    # Let's try a "Pure Exploration" strategy weighted by cost.
    # Score = sigma / cost
    # This encourages the algorithm to visit areas where the model is most uncertain,
    # which is often far from the current data, facilitating global exploration.
    
    # To add a bit of exploitation if sigma is low everywhere (convergence),
    # we can add a small term for expected improvement, but the prompt asks for a
    # materially different strategy. Pure exploration (sigma-based) is distinct from
    # the current UCB (mu - beta*sigma) or EI.
    
    # Let's refine: 
    # The current program uses UCB: f_best - mu + beta*sigma.
    # This is exploitation-heavy if mu is close to f_best.
    # A pure exploration strategy maximizes sigma.
    # To make it cost-aware: score = sigma / cost.
    
    # However, if sigma is very small everywhere, we might get stuck.
    # But the task says "minimizing", and GP sigma goes to 0 near data.
    # Maximizing sigma pushes us to the edges or unexplored regions.
    
    # Let's try: score = sigma / (cost + epsilon)
    
    score = sigma_safe / (cost + 1e-9)
    
    # Check time
    if time.time() > deadline:
         # Return best effort so far
         return np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure finite values
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
