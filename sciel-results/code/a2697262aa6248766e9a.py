import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Handle zero sigma to avoid issues, though TS handles sigma=0 naturally (deterministic)
    sigma_safe = np.maximum(sigma, 0.0)
    
    # Cost is in [1, 2] approximately, but ensure no division by zero
    cost_safe = np.maximum(cost, 1e-12)

    # Check deadline early
    if time.time() > deadline:
        # Fallback to a simple greedy cost-aware score if running out of time
        # Use -mu / cost as a deterministic fallback
        scores = -mu / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    # Thompson Sampling:
    # Sample from the posterior: f_sampled = mu + sigma * z
    # z ~ N(0, 1)
    # Note: Using np.random.standard_normal ensures we get a fresh sample each call,
    # which is the correct behavior for stochastic Thompson Sampling in a BO loop.
    
    # If we want to be more robust against the specific random state of the evaluator,
    # we could consider if a fixed seed is better, but standard TS is stochastic.
    # Given the "stagnation" directive, stochasticity helps escape local optima.
    
    z = np.random.standard_normal(size=mu.shape)
    
    # Sampled objective values
    f_sampled = mu + sigma_safe * z
    
    # Cost-aware score:
    # We want to minimize f_sampled / cost.
    # Since the acquisition function returns scores where HIGHER is better,
    # we negate the ratio.
    scores = -f_sampled / cost_safe
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Final deadline check
    if time.time() > deadline:
        return np.where(np.isfinite(scores), scores, 0.0)
        
    return scores
