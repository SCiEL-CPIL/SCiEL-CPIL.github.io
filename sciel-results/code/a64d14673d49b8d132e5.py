import numpy as np
import time

# Global start time for deadline checking
_START_TIME = time.time()
_DEADLINE = 310.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Args:
        mu: Array of GP means (shape 2000,).
        sigma: Array of GP standard deviations (shape 2000,).
        f_best: Scalar, lowest observed value.
        cost: Array of evaluation costs (shape 2000,).
        
    Returns:
        Array of scores (shape 2000,). Higher scores are preferred.
    """
    # Check deadline - if we're close to the limit, just return a quick result
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE:
        # Emergency fallback: return deterministic scores based on -mu/cost
        cost_safe = np.maximum(cost, 1e-6)
        scores = -mu / cost_safe
        scores = np.where(np.isfinite(scores), scores, -np.inf)
        return scores
    
    # Thompson Sampling: sample a single theta from N(0, 1)
    # For each candidate, the sampled value is mu[i] + sigma[i] * theta
    # We want to minimize, so we score by -(sampled value) / cost
    # Higher score = better (lower objective value per unit cost)
    
    # Sample theta from standard normal
    theta = np.random.randn()
    
    # Compute sampled objective values
    sampled_values = mu + sigma * theta
    
    # Cost-aware scoring: we want low objective value and low cost
    # Score = -sampled_value / cost (higher is better)
    cost_safe = np.maximum(cost, 1e-6)
    scores = -sampled_values / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
