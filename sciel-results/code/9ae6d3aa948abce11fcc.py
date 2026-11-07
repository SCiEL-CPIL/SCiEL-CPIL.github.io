import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling for Bayesian optimization.
    """
    # Ensure inputs are 1D arrays
    mu = np.asarray(mu).flatten()
    sigma = np.asarray(sigma).flatten()
    
    # Sanitize sigma to be positive and finite
    sigma_safe = np.maximum(sigma, 1e-8)
    sigma_safe = np.where(np.isfinite(sigma_safe), sigma_safe, 1e-8)
    
    # Sanitize mu to be finite
    mu_safe = np.where(np.isfinite(mu), mu, 0.0)
    
    # Generate random samples from the posterior distribution
    # z ~ N(0, 1)
    z = np.random.randn(len(mu_safe))
    
    # Sample from N(mu, sigma^2)
    samples = mu_safe + sigma_safe * z
    
    # The acquisition score is the negative of the sampled value
    # because we want to maximize the score (which corresponds to minimizing the function)
    score = -samples
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
