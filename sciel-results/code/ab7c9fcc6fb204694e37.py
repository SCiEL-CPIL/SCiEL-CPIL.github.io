"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for time management and random seeding
_start_time = time.time()
_timeout = 330.0  # Evaluator kills after 330s, leave safety margin
_rng = np.random.default_rng()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    mu, sigma, f_best, cost: arrays of shape (2000,)
    Returns: array of shape (2000,) with finite scores (higher is better)
    """
    # Time guard: if we are getting close to the limit, return a safe deterministic fallback
    # to ensure we don't crash or exceed time. However, since this function is called
    # once per iteration and is very fast, the main risk is if the *loop* in the evaluator
    # is slow, but we only control this function. 
    # We check elapsed time just in case this function is called in a tight loop elsewhere 
    # or if the overhead is unexpected.
    if time.time() - _start_time > _timeout - 1.0:
        # Fallback: return something stable, e.g., just -mu/cost (greedy exploitation)
        # This ensures we return finite numbers quickly.
        return -mu / (cost + 1e-9)

    # Thompson Sampling: Draw a sample from the posterior
    # The posterior is Normal(mu, sigma^2).
    # We want to maximize the sampled value.
    # However, we must account for cost.
    
    # To make the exploration stochastic but consistent within the call,
    # we generate noise.
    # Note: The evaluator calls this once per iteration. 
    # The randomness comes from the Gaussian noise added to the mean.
    
    # Ensure sigma is positive to avoid division issues, though GP sigma should be > 0
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Generate random noise
    # We use a single random number per point
    noise = _rng.standard_normal(size=mu.shape)
    
    # Sample from posterior: mu + sigma * noise
    sample = mu + sigma_safe * noise
    
    # Cost-aware: Divide by cost
    # Higher score means we evaluate next.
    # We want to maximize sample / cost.
    # Since sample can be negative (Rosenbrock values are >= 0, but GP predictions might be negative due to noise/standardization artifacts),
    # we just use the ratio.
    
    # Handle division by zero in cost (cost is 1 + mean(x_scaled), so min cost is 1.0, so safe)
    scores = sample / cost
    
    # Ensure finiteness
    # If any NaN/Inf appears, replace with a very low value (since we want to maximize)
    scores = np.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
    
    return scores
# EVOLVE-BLOCK-END
