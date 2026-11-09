import numpy as np
import time

# Global state
_STEP = 0
_TOTAL_STEPS = 25  # 30 total evals - 5 initial

# Wall-clock deadline guard
_T0 = time.time()
_WALL_LIMIT = 330.0
_SAFETY = 45.0
_DEADLINE = _WALL_LIMIT - _SAFETY

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for minimizing Hartmann 6d.
    
    Instead of using a fixed or adaptive exploration constant (as in UCB),
    we sample a single value from the posterior Gaussian for each candidate
    and return the negative of that sample. The candidate with the minimum
    sampled value gets the highest score. This naturally balances exploration
    and exploitation without manual tuning of exploration constants.
    
    Args:
        mu: (2000,) array of posterior means.
        sigma: (2000,) array of posterior standard deviations.
        f_best: float, the lowest observed value.
        
    Returns:
        (2000,) array of scores. Higher score means evaluate next.
    """
    global _STEP
    
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback to simple greedy if time is running out
        return -mu
    
    # Update step counter
    _STEP += 1
    
    # Thompson Sampling: sample from N(mu, sigma^2) for each candidate
    # We want to minimize f, so we want the candidate with the smallest sampled value
    # to have the highest score. So score = -sampled_value.
    
    # Generate random normal samples
    rng = np.random.default_rng()
    eps = rng.standard_normal(len(mu))
    
    # Sampled values from the posterior
    samples = mu + sigma * eps
    
    # Score: higher is better, so we negate the sampled value
    scores = -samples
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
