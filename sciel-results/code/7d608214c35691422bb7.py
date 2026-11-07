import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    Args:
        mu: Array of mean predictions (2000,).
        sigma: Array of standard deviation predictions (2000,).
        f_best: The best (lowest) function value observed so far.
        
    Returns:
        Array of scores (2000,). Higher scores indicate better candidates.
    """
    # Ensure sigma is strictly positive to prevent division by zero or invalid normal distribution parameters
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Sample from the posterior predictive distribution N(mu, sigma^2)
    # np.random.normal broadcasts the loc and scale arrays
    samples = np.random.normal(loc=mu, scale=sigma_safe)
    
    # The goal is to minimize the function value.
    # The evaluator selects the candidate with the highest score.
    # Therefore, we return the negative of the sampled values.
    # Maximizing -sample is equivalent to minimizing sample.
    return -samples
