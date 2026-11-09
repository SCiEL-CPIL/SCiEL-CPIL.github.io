import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    Draws a single sample from the posterior distribution N(mu, sigma^2) 
    and returns the negative sampled values as scores.
    This provides a stochastic exploration-exploitation balance suitable 
    for multimodal functions like Goldstein-Price 2D.
    """
    # Ensure arrays are floats and handle edge cases for sigma
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is non-negative and finite
    sigma = np.maximum(sigma, 1e-8)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Generate standard normal noise
    # We use a fixed seed logic? No, Thompson sampling relies on randomness.
    # However, in a stateless function call, we just draw fresh noise each time.
    # The evaluator calls this function in a loop. Each call represents a step.
    # Drawing fresh noise each call is the correct TS behavior.
    z = np.random.standard_normal(size=mu.shape)
    
    # Sample from the posterior
    # f_sample = mu + sigma * z
    # We want to minimize f, so we maximize -f_sample
    scores = -(mu + sigma * z)
    
    # Ensure output is finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
