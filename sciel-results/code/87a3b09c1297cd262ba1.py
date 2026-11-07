import numpy as np

def acquisition(mu, sigma, f_best, cost):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Use a local random generator for cleaner state management
    rng = np.random.default_rng()
    
    # Sample objective values from the posterior
    samples = rng.normal(loc=mu, scale=sigma)
    
    # Standard Cost-Effective Thompson Sampling score
    # We want to maximize the probability of improvement per unit cost.
    # Improvement is f_best - sample.
    # Score is (f_best - sample) / cost.
    
    # To improve upon pure TS, we add a deterministic exploitation term.
    # This term rewards candidates where the posterior mean is already better than f_best.
    # This reduces the variance of the selection process, making it more robust.
    
    # Calculate the stochastic part
    stochastic_part = (f_best - samples) / cost
    
    # Calculate the deterministic exploitation bias
    # Weighted by 0.5 to balance exploration and exploitation
    exploitation_bias = 0.5 * (f_best - mu) / cost
    
    # Combine
    scores = stochastic_part + exploitation_bias
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
