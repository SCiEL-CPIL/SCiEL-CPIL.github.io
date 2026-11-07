"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline management
_START_TIME = time.time()
_DEADLINE_SECONDS = 290.0  # Safety margin before 300s/330s limit

def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_SECONDS:
        # Return a safe, finite, valid score to avoid invalidation
        # Use a neutral score (e.g., all zeros) to not break the loop
        return np.zeros_like(mu, dtype=np.float64)
    
    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma and cost are strictly positive to avoid division errors
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Thompson Sampling: Sample from the posterior distribution for each candidate
    # This provides a balance between exploration (high sigma) and exploitation (low mu)
    # which is often superior to deterministic EI for multimodal functions like Rastrigin.
    
    # Generate standard normal random variables
    z = np.random.randn(mu.shape)
    
    # Sample from N(mu, sigma)
    samples = mu + sigma * z
    
    # We want to minimize the objective.
    # Therefore, a lower sampled value is better.
    # We also want to account for cost: cheaper points are preferred.
    # Score = -sampled_value / cost
    # Higher score means (lower value) / (lower cost), which is desirable.
    
    scores = -samples / cost
    
    # Handle any potential numerical issues (NaNs or Infs)
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
