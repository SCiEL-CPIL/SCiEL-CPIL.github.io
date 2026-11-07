"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard to ensure we finish well before the 330s limit
_start_time = time.time()
_DEADLINE = 320.0  # 10s safety margin

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    Instead of scoring each candidate with a deterministic acquisition function 
    (like EI or UCB), we sample a function value from the GP posterior for each 
    candidate. The posterior at each point x is a Normal distribution with mean 
    mu[x] and standard deviation sigma[x].
    
    We draw a single sample for each of the 2000 candidates and return the negative 
    of the sampled value as the score. The candidate with the highest score 
    (lowest sampled value) is selected for the next evaluation.
    
    This naturally balances exploration (high sigma leads to wider sample range, 
    potentially lower values) and exploitation (low mu leads to lower sample values).
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback to a simple deterministic strategy if we are running out of time
        # This is extremely unlikely given the trivial computation, but good practice.
        return -mu
    
    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Draw a single sample from the posterior N(mu, sigma^2) for each candidate
    # We use a single random draw for each point. 
    # To make it deterministic per call if needed, we could seed, but 
    # the task implies standard stochastic BO where the randomness is part of the method.
    # However, the evaluator runs the function once per step. 
    # Standard Thompson Sampling draws a sample from the posterior predictive distribution.
    
    # Generate random normal variables
    z = np.random.standard_normal(len(mu))
    
    # Sampled function values
    samples = mu + sigma_safe * z
    
    # We want to minimize the function, so we want the lowest sampled value.
    # The acquisition score should be higher for better candidates.
    # So we return the negative of the sampled value.
    scores = -samples
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
