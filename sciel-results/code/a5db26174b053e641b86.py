import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Args:
        mu: Array of posterior means (shape 2000,)
        sigma: Array of posterior standard deviations (shape 2000,)
        f_best: The best function value observed so far (scalar)
        
    Returns:
        Array of scores (shape 2000,), where higher scores indicate better candidates.
    """
    # Draw samples from the posterior predictive distribution
    # y ~ N(mu, sigma^2)
    # We use z ~ N(0, 1) so y = mu + sigma * z
    
    z = np.random.standard_normal(len(mu))
    
    # Compute sampled values
    sampled_values = mu + sigma * z
    
    # We are minimizing the function, so we want the smallest sampled_values.
    # The evaluator picks the candidate with the highest score.
    # Therefore, we return the negative of the sampled values.
    scores = -sampled_values
    
    return scores
