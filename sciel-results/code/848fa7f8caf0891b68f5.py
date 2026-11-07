"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_deadline = _start_time + 290.0

def acquisition(mu, sigma, f_best):
    # Deadline guard: if we are running out of time, fall back to greedy (minimize mu)
    if time.time() > _deadline:
        return -mu
    
    # Ensure sigma is strictly non-negative and finite
    sigma = np.maximum(sigma, 0.0)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure mu is finite
    mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Structural Change: Normalized UCB (N-UCB)
    # Standard UCB: score = -mu + beta * sigma
    # N-UCB normalizes sigma by the mean sigma of the current batch.
    # This highlights points with uncertainty significantly higher than the average,
    # promoting exploration of novel regions relative to the current candidate set.
    
    # Calculate mean sigma for normalization
    sigma_mean = np.mean(sigma)
    
    # Normalize sigma: values > mean get score > 0.5, values < mean get < 0.5
    # Add small epsilon to avoid division by zero if all sigmas are 0
    norm_sigma = sigma / (sigma + sigma_mean + 1e-6)
    
    # Beta coefficient: 
    # Start with high exploration (beta > 1) and decay as we get closer to the optimum.
    # f_best is the lowest observed value. Ackley min is 0.
    # If f_best is large, we are far from optimum -> explore more.
    # If f_best is small, we are close -> exploit more.
    
    # Adaptive beta: ranges from ~2.0 (far) to ~0.5 (close)
    # Using a simple decay function based on f_best
    beta = 1.0 + 1.0 * np.exp(-0.5 * f_best)
    
    # Combine exploitation (-mu) and normalized exploration (norm_sigma)
    scores = -mu + beta * norm_sigma
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
