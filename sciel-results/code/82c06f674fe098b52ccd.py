"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 325.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    This function uses Thompson Sampling by drawing a sample from the GP posterior
    and dividing by the evaluation cost. This introduces stochastic exploration
    that can escape local optima more effectively than deterministic EI.
    
    Args:
        mu: array of shape (2000,) - GP mean predictions
        sigma: array of shape (2000,) - GP standard deviation predictions
        f_best: float - lowest observed function value
        cost: array of shape (2000,) - cost of evaluating each candidate
        
    Returns:
        array of shape (2000,) - acquisition scores (higher is better)
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a valid finite array to avoid invalidating the result
        return np.random.uniform(size=mu.shape)
    
    # Ensure sigma is positive for numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Draw a random sample from the GP posterior: f(x) ~ N(mu, sigma^2)
    # This is the core of Thompson Sampling
    eps = np.random.randn(mu.shape[0])
    posterior_sample = mu + sigma_safe * eps
    
    # For minimization, we want to find low values
    # Thompson Sampling for minimization: minimize the posterior sample
    # But we need to return scores where higher = evaluate next
    # So we negate the sample: score = -posterior_sample / cost
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    cost_safe = np.maximum(cost, 1e-8)
    scores = -posterior_sample / cost_safe
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
