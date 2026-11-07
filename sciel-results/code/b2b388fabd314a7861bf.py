import numpy as np

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Args:
        mu: Mean of GP posterior, shape (2000,)
        sigma: Std dev of GP posterior, shape (2000,)
        f_best: Best observed value (scalar)
        cost: Cost of evaluating each candidate, shape (2000,)
        
    Returns:
        scores: Shape (2000,), higher is better.
    """
    # Ensure sigma is strictly positive to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Generate standard normal random variables
    z = np.random.randn(len(mu))
    
    # Sample from the posterior distribution
    samples = mu + sigma_safe * z
    
    # We want to minimize the function value f(x).
    # Thompson Sampling suggests picking the x that minimizes the sampled value.
    # To use a maximization-based acquisition (higher score = pick), we negate the sample.
    # We also divide by cost to make it cost-aware.
    scores = -samples / cost
    
    # Replace any non-finite values with -inf to ensure they are never picked
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
