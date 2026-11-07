"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state for deadline guard
_bo_start_time = None
_bo_call_count = 0

def acquisition(mu, sigma, f_best):
    global _bo_start_time, _bo_call_count
    
    # Initialize start time on first call
    if _bo_start_time is None:
        _bo_start_time = time.time()
    
    _bo_call_count += 1
    
    # Deadline Guard: 330s limit, use 305s as safety margin
    if time.time() - _bo_start_time > 305:
        # If we are close to timeout, prioritize speed and exploitation
        # to ensure the loop finishes.
        return -mu
    
    # Convert to numpy arrays for safety
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Thompson Sampling approach:
    # Sample from the posterior at each candidate point, then score by
    # the negative of the sampled value. This provides stochastic exploration
    # that naturally balances exploration and exploitation based on the
    # posterior uncertainty without explicit tuning parameters.
    
    # Generate random samples from the posterior
    # For each point, sample from N(mu_i, sigma_i^2)
    rng = np.random.RandomState(_bo_call_count * 12345)
    samples = mu + sigma * rng.randn(len(mu))
    
    # Score: higher is better, so we want low sampled values
    # Use negative of the sample as the score
    scores = -samples
    
    # Add a small exploration bonus for points with high uncertainty
    # This helps escape local optima in multimodal functions like Ackley
    # Scale the uncertainty bonus to be comparable to the function values
    sigma_scale = np.max(sigma) if np.max(sigma) > 1e-12 else 1.0
    exploration_bonus = 0.3 * sigma / sigma_scale
    
    scores = scores + exploration_bonus
    
    # Ensure finite values
    scores[~np.isfinite(scores)] = 0.0
    
    return scores
# EVOLVE-BLOCK-END
