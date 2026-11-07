import numpy as np

def acquisition(mu, sigma, f_best):
    # Thompson Sampling approach
    # The GP posterior provides mu and sigma for 2000 candidates.
    # We sample a single value from the posterior predictive distribution for each candidate.
    # The candidate with the lowest sampled value is the most promising minimizer.
    # Since the evaluator selects the candidate with the highest score,
    # we return the negative of the sampled values.
    
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-8)
    
    # Draw standard normal random numbers
    z = np.random.randn(mu.shape[0])
    
    # Sample from N(mu, sigma^2)
    samples = mu + sigma * z
    
    # Return negative samples to maximize the score for the minimum value
    return -samples
