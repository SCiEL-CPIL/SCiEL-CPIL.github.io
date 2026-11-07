"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 10s safety margin before 330s limit

def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    if time.time() > _DEADLINE:
        return np.zeros(len(mu))

    # Thompson Sampling acquisition
    # Draw a sample from the posterior: f_sample = mu + sigma * z
    # We want to minimize f, so we want low f_sample.
    # Score should be high for candidates we want to evaluate.
    # So score = -f_sample / cost.
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # Draw standard normal noise
    z = np.random.randn(len(mu))
    
    # Sample from posterior
    f_sample = mu + sigma * z
    
    # Cost-aware score
    cost_safe = np.maximum(cost, 1.0)
    
    # We want to minimize f, so higher score means lower f_sample / cost
    # Using negative of the sampled value makes it a maximization problem
    scores = -f_sample / cost_safe
    
    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
