import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    Uses Thompson Sampling to balance exploration and exploitation.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is strictly positive to avoid division by zero or numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling: Sample from the posterior distribution of the function values.
    # We draw a single noise vector z ~ N(0, 1) for all candidates.
    # The sampled value is y = mu + sigma * z.
    # We want to minimize the function, so we maximize the score -y.
    # score = -(mu + sigma * z) = -mu - sigma * z
    
    # Generate standard normal noise
    z = np.random.randn(len(mu))
    
    scores = -mu - sigma * z
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
