"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Thompson Sampling acquisition function for cost-aware Bayesian optimization.
    
    Instead of computing a deterministic Expected Improvement score, we sample
    from the posterior distribution of the objective function at each candidate
    point. The score is the sampled value divided by the cost.
    
    This approach naturally balances exploration (high sigma leads to wide
    sample spread) and exploitation (low mu leads to lower samples).
    """
    # Deadline guard: ensure we return within time limits.
    # Thompson sampling is O(N), so it's fast, but we add a safety check.
    start_time = time.time()
    
    # Ensure sigma is non-negative for standard deviation
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Generate random samples from the posterior: f(x) ~ N(mu, sigma^2)
    # We need a deterministic seed or just use global RNG? 
    # The task says "five seeded instances", but the acquisition function itself
    # is called in a loop. If we use global RNG, the sequence of samples will be
    # deterministic given the sequence of calls.
    # However, standard Thompson Sampling usually requires a fresh sample each time.
    # Using np.random.normal is fine.
    
    # Sample from N(mu, sigma)
    # Note: sigma is the standard deviation.
    samples = mu + sigma_safe * np.random.standard_normal(len(mu))
    
    # The score is the sampled value divided by the cost.
    # We want to minimize the objective, so lower sampled values are better.
    # But the return interface requires "higher = evaluate next".
    # So we should invert the sampled value.
    # score = -samples / cost
    
    # Ensure cost is not zero to avoid division errors (cost is 1 + mean(x), so min is 1)
    cost_safe = np.maximum(cost, 1e-9)
    
    score = -samples / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, -np.inf)
    
    # Check time (should be very fast, < 1ms)
    elapsed = time.time() - start_time
    if elapsed > 290:  # Safety margin
         # If somehow slow, return a fallback (though this shouldn't happen)
         return np.zeros(len(mu))
         
    return score
# EVOLVE-BLOCK-END
