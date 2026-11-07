import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Args:
        mu: Array of mean predictions (shape: N,)
        sigma: Array of standard deviation predictions (shape: N,)
        f_best: Current best observed value (scalar)
        
    Returns:
        Array of scores (shape: N,). Higher scores indicate candidates 
        that should be evaluated next.
    """
    # Ensure sigma is non-negative and has a minimum floor to avoid division by zero 
    # or numerical issues, although GP sigma is typically > 0.
    # We don't divide, but we multiply, so 0 is fine, but tiny values are safer.
    safe_sigma = np.maximum(sigma, 1e-9)
    
    # Draw random samples from the standard normal distribution
    # This uses the global numpy random state, which is expected to be seeded 
    # by the evaluator for reproducibility across seeds.
    z = np.random.standard_normal(len(mu))
    
    # Compute the sampled objective values
    # y ~ N(mu, sigma^2)
    samples = mu + safe_sigma * z
    
    # The goal is to minimize the objective function.
    # The evaluator selects the candidate with the highest acquisition score.
    # Therefore, we return the negative of the sampled values.
    # This ensures that candidates with lower sampled values (better objectives)
    # receive higher scores.
    return -samples
