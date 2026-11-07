import numpy as np

def acquisition(mu, sigma, f_best):
    # Thompson Sampling: sample from GP posterior and return negative of sampled values
    # This directly optimizes for the expected minimum value
    # We use a single sample approach which is computationally efficient
    
    # Clamp f_best to [0, 25] to be safe
    f_best_clamped = np.clip(f_best, 0.0, 25.0)
    
    # For Thompson Sampling, we sample from the posterior:
    # x ~ N(mu, sigma^2)
    # We want to minimize f(x), so we return -x as the score (higher is better)
    
    # Generate random samples from the GP posterior
    # Use a single sample for efficiency
    samples = mu + sigma * np.random.randn(len(mu))
    
    # Return negative of the sampled values (we want to maximize -f)
    score = -samples
    
    return score
