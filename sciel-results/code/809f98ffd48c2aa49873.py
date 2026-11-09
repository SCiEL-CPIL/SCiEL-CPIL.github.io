import numpy as np
import time
from scipy.stats import norm

# Global state for adaptive beta and time tracking
_start_time = None
_last_cost = 0.0

def acquisition(mu, sigma, f_best, cost):
    global _start_time, _last_cost
    
    # Initialize time tracking
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    
    # Safety margin: if we are close to timeout, return something safe
    # The evaluator kills at 330s. We should finish fast.
    # This function is called every iteration. It must be fast.
    
    # Calculate total cost so far
    total_cost = np.sum(cost)
    
    # Adaptive beta:
    # Early on (low cost), we want high exploration (high beta).
    # Later (high cost), we want exploitation (low beta).
    # Cost budget is 30.
    # Let's decay beta exponentially with total cost.
    # beta = base * exp(-decay * total_cost)
    # At start, total_cost ~ 5 (5 initial points). 
    # At end, total_cost ~ 30.
    
    # We want beta to be large enough to explore sigma.
    # Typical sigma might be in the range of objective function values.
    # Powell 4d min is 0. Values can be large.
    # Let's use a beta that scales with the current uncertainty or just a fixed schedule.
    # A common choice is beta = 2 * log(t) or similar, but here we have cost.
    
    # Let's try: beta = 5.0 * np.exp(-0.1 * total_cost)
    # At total_cost=5, beta ~ 5 * exp(-0.5) ~ 3.0
    # At total_cost=30, beta ~ 5 * exp(-3.0) ~ 0.25
    
    beta = 5.0 * np.exp(-0.1 * total_cost)
    
    # LCB: mu - beta * sigma
    # We want to MINIMIZE this value.
    # The acquisition function returns a score where HIGHER is better.
    # So we return -LCB.
    # But we also need to account for cost efficiency.
    # The problem says "Cost-aware acquisition".
    # Dividing by cost encourages low-cost evaluations.
    # However, LCB is already a balance of mean and variance.
    # Dividing by cost might distort the balance if sigma is large.
    # Let's try: score = - (mu - beta * sigma) / cost
    
    # One issue: sigma can be very small in explored regions, leading to huge scores if mu is low.
    # This is good for exploitation.
    # In unexplored regions, sigma is large, so - (mu - beta*sigma) is high (if beta*sigma > mu).
    # This encourages exploration.
    
    # To ensure finite scores and avoid numerical issues:
    sigma_safe = np.maximum(sigma, 1e-6)
    
    lcb = mu - beta * sigma_safe
    score = -lcb / cost
    
    # Add a tiny bit of noise to break ties? No, deterministic is better for reproducibility.
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
