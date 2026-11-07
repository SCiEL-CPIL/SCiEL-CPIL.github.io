"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Thompson Sampling: Sample from the posterior predictive distribution
    # mu and sigma are in objective units
    # cost has shape (2000,)
    
    # Generate standard normal samples
    # Use a fixed seed? No, stochasticity is the point of TS.
    # However, for reproducibility in debugging, one might use a seed, 
    # but the task implies a stochastic process is allowed/expected for TS.
    # The evaluator runs 5 instances. We should use the global RNG or a local one.
    # Using np.random.normal is standard.
    
    eps = np.random.normal(0, 1, size=mu.shape)
    
    # Sampled objective values
    y_sampled = mu + sigma * eps
    
    # We want to minimize y_sampled / cost
    # So we maximize -y_sampled / cost
    
    # Handle potential division by zero or very small costs
    # Cost is 1 + mean(x_scaled), so cost >= 1. No division by zero risk.
    
    scores = -y_sampled / cost
    
    # Deadline guard
    # The task says "Each evaluation must finish within 300 s".
    # This function is part of that evaluation.
    # If this function takes too long, it contributes to the timeout.
    # Vectorized numpy ops on 2000 floats are microseconds.
    # But we check just in case.
    if time.time() - start_time > 10:  # 10s is way more than needed, but safe
        # Return a fallback, e.g., based on mean only
        return -mu / cost

    return scores
# EVOLVE-BLOCK-END
