import numpy as np

def acquisition(mu, sigma, f_best):
    # Thompson Sampling
    # Draw a sample from the posterior distribution at each candidate
    # f(x) ~ N(mu(x), sigma(x)^2)
    # We want to minimize f, so we assign a score of -f_sample
    # Higher score means lower sampled function value
    
    # Ensure sigma is non-negative and handle zeros
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Generate standard normal random variables
    z = np.random.normal(0, 1, size=mu.shape)
    
    # Sample from the posterior
    samples = mu + sigma_safe * z
    
    # Return negative samples as scores
    # The evaluator selects the candidate with the highest score
    scores = -samples
    
    return scores
