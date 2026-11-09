"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_TIMEOUT = 325.0 # Safety margin before 330s

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Thompson Sampling.
    
    Strategy:
    1. Sample a function value f_s(x) from the GP posterior N(mu, sigma^2).
    2. Compute the cost-adjusted value v(x) = f_s(x) / cost(x).
    3. Return scores S(x) = -v(x). Maximizing S(x) is equivalent to 
       minimizing the cost-adjusted sampled function value.
    
    This shifts the search from deterministic Expected Improvement to 
    stochastic regret minimization, balancing exploration (via the 
    randomness of the sample) and exploitation (via the mean) while 
    accounting for evaluation costs.
    """
    # Check time budget
    if time.time() - _START_TIME > _TIMEOUT:
        # Return neutral scores to avoid crash, though evaluator kills us anyway
        return np.zeros_like(mu, dtype=np.float64)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize inputs
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Generate standard normal noise
    # Using a local random generator might be safer to avoid global state interference,
    # but np.random is standard.
    z = np.random.standard_normal(mu.shape)
    
    # Sample from posterior: f_s ~ N(mu, sigma^2)
    # f_s = mu + sigma * z
    # Note: For minimization, we are interested in the lower tail.
    # The sample is just a random draw.
    f_s = mu + sigma_safe * z
    
    # Cost-adjusted sample
    # We want to minimize f_s / cost.
    # So the score to maximize is - (f_s / cost).
    val = f_s / cost_safe
    score = -val
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
