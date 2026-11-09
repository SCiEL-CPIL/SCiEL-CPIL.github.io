import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Args:
        mu: Array of shape (N,) with mean predictions.
        sigma: Array of shape (N,) with standard deviations.
        f_best: Scalar, the best observed function value so far.
        
    Returns:
        scores: Array of shape (N,) where higher values indicate better candidates.
    """
    # Ensure sigma is strictly positive to avoid numerical issues in sampling
    # The GP noise is 1e-6, so sigma should be at least sqrt(1e-6) ~ 1e-3,
    # but we guard against zero or negative values just in case.
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Sample from the posterior distribution N(mu, sigma^2)
    # We are minimizing, so we want the candidate with the lowest sampled value.
    # The evaluator selects the candidate with the HIGHEST score.
    # Therefore, we return the negative of the sampled value.
    samples = np.random.normal(loc=mu, scale=sigma_safe)
    scores = -samples
    
    # Ensure all scores are finite to satisfy the contract
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
