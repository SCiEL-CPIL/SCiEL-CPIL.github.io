"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

global_total_cost = 0.0
global_call_count = 0
global_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global global_total_cost
    global global_call_count
    global global_start_time
    
    # Initialize time on first call
    if global_start_time is None:
        global_start_time = time.time()
    else:
        # Safety deadline guard
        elapsed = time.time() - global_start_time
        if elapsed > 290:  # Safety margin before 300s
             # Return a default result, e.g., exploitation
             return -mu

    # Estimate total cost
    # Initial 5 points cost ~ 7.5
    # Each subsequent point costs ~ 1.5
    est_total_cost = 7.5 + 1.5 * global_call_count
    
    # Adaptive exploration-exploitation trade-off
    # Decay exploration bonus with total cost
    base_kappa = 0.5
    decay_rate = 0.1
    kappa = base_kappa * np.exp(-decay_rate * est_total_cost)
    
    # Calculate Expected Improvement
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add exploration bonus
    score = ei + kappa * sigma
    
    # Cost-aware scaling
    score = score / cost
    
    # Update state
    global_call_count += 1
    
    return score

# EVOLVE-BLOCK-END
