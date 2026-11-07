import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for minimization.
    
    Parameters:
    mu: array of shape (2000,) with posterior mean.
    sigma: array of shape (2000,) with posterior standard deviation.
    f_best: scalar, the best value observed so far (not directly used in TS scoring).
    
    Returns:
    array of shape (2000,) with scores. Higher score means more likely to be selected.
    """
    # Draw samples from the posterior distribution N(mu, sigma^2)
    # We use standard normal random variables z
    z = np.random.randn(len(mu))
    
    # Sampled values
    samples = mu + sigma * z
    
    # Since we are minimizing, we want the candidate with the lowest sampled value.
    # The evaluator selects the candidate with the HIGHEST score.
    # Therefore, we return the negative of the sampled values.
    return -samples
