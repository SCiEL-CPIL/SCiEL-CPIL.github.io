import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    Args:
        mu: Mean of the GP posterior at candidate points. Shape (N,).
        sigma: Standard deviation of the GP posterior at candidate points. Shape (N,).
        f_best: The best (lowest) observed function value so far.
        
    Returns:
        scores: Scores for each candidate. Higher score indicates a better candidate.
    """
    # Thompson Sampling: Sample from the posterior distribution
    # y ~ N(mu, sigma^2)
    # We want to minimize y, so we return -y as the score.
    # This encourages exploration in areas with high uncertainty (high sigma)
    # and exploitation in areas with low predicted values (low mu).
    
    # Generate standard normal random variables
    z = np.random.randn(len(mu))
    
    # Sample from the posterior
    y = mu + sigma * z
    
    # Return negative sample as score (higher is better for minimization)
    return -y
