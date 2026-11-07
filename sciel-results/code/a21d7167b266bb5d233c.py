"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track time and seed for reproducibility within the process
_start_time = time.time()
_deadline = _start_time + 320  # Safety margin before 330s limit
_rng = np.random.default_rng(42)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Args:
        mu: (2000,) array of GP means
        sigma: (2000,) array of GP standard deviations
        f_best: float, best observed value
        cost: (2000,) array of costs for each candidate
        
    Returns:
        (2000,) array of acquisition scores (higher is better)
    """
    # Check deadline
    if time.time() > _deadline:
        # Fallback: return a simple heuristic to avoid timeout/invalidity
        # Prefer low cost and low mean
        return -mu / (cost + 1e-9)
    
    # Clip sigma to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling: Draw a sample from N(mu, sigma^2) for each candidate
    # We draw a single random vector z ~ N(0, 1) and compute sample = mu + sigma * z
    # This is efficient and ensures correlation structure is handled by the GP
    z = _rng.standard_normal(len(mu))
    samples = mu + sigma * z
    
    # For minimization, we want low values.
    # To make "higher score = better", we invert the sample.
    # Also normalize by cost to account for acquisition cost.
    # Score = -sample / cost
    # To ensure finite and stable, we add a small epsilon to cost
    scores = -samples / (cost + 1e-9)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
