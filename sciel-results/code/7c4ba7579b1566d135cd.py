import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    Uses Thompson Sampling: sample from the posterior N(mu, sigma^2) and
    return the negative sample as the score (higher score = lower sampled value = better).
    This provides natural exploration-exploitation balance without manual beta tuning.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 2.0
    
    # Thompson Sampling: draw one sample from the posterior
    # f ~ N(mu, sigma^2)
    # We want to minimize f, so higher score should correspond to lower sampled f
    # Score = -sampled_f, so argmax score = argmin sampled_f
    
    # Use a fixed random seed per call? No, we want stochasticity.
    # But we need reproducibility? The task doesn't require it, just valid scores.
    # Generate standard normal random numbers
    z = np.random.standard_normal(mu.shape)
    
    # Sample from the posterior
    sampled_f = mu + sigma * z
    
    # Score: higher is better for the acquisition
    # Since we're minimizing, we want to pick the point with the lowest sampled value
    # So score = -sampled_f
    scores = -sampled_f
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    # Check deadline
    if time.time() > deadline:
        return scores
    
    return scores
