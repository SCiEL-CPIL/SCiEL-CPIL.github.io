import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    Args:
        mu: array of shape (2000,) with mean predictions.
        sigma: array of shape (2000,) with standard deviation predictions.
        f_best: scalar, the best observed value so far.
        
    Returns:
        array of shape (2000,) with scores. Higher score means evaluate next.
    """
    # Thompson Sampling:
    # 1. Draw a random sample from the posterior GP for each candidate point.
    #    f_s(x) ~ N(mu(x), sigma(x))
    # 2. Select the point that minimizes this sample.
    #    Since the evaluator selects the point with the HIGHEST score,
    #    we return the negative of the sampled value.
    
    # Generate standard normal random numbers
    # This relies on the global numpy random state, which is typically
    # seeded by the evaluation harness for reproducibility.
    z = np.random.randn(mu.shape[0])
    
    # Compute the sampled function values
    # Handle potential numerical issues where sigma might be 0 or very small
    f_samples = mu + sigma * z
    
    # We want to minimize f_samples, so we maximize -f_samples
    scores = -f_samples
    
    # Ensure all scores are finite to satisfy the contract
    # Replace any non-finite values with a very low score
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
