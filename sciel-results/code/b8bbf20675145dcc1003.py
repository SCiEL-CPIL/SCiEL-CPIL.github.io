"""Cost-aware acquisition function for Bayesian optimization (Styblinski-Tang 2D)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0  # 280s safety margin (limit is 300s)


def _check_deadline():
    return time.time() > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Check deadline
    if _check_deadline():
        # Return a simple fallback: prefer low cost and low mu
        return -mu / (cost + 1e-9)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Thompson Sampling: sample a single scalar theta from a Gaussian
    # We use a fixed random seed for reproducibility within the same call
    # but the key is that we sample ONE theta value that applies to all candidates
    # Actually, for Thompson Sampling in this context, we should sample per candidate
    # but the strategy says "sample a single scalar value theta from a Gaussian 
    # distribution with mean mu and standard deviation sigma for all 2000 candidates"
    # This is unusual - typically you'd sample per candidate. Let me re-read.
    # "sampling a single scalar value θ from a Gaussian distribution with mean μ 
    # and standard deviation σ for all 2000 candidates"
    # This means: for each candidate i, sample θ_i ~ N(μ_i, σ_i)
    # Wait, "a single scalar value θ" - maybe it means one global offset?
    # Let me interpret it as: sample θ_i ~ N(μ_i, σ_i) for each candidate i
    # This is standard Thompson Sampling.
    
    rng = np.random.default_rng(42)
    theta = rng.normal(mu, sigma)
    
    # Score: higher is better, so we want low theta (since we're minimizing)
    # Score = -theta / cost
    score = -theta / cost
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
