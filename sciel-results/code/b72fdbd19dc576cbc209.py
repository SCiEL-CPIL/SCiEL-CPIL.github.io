import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function.
    
    Instead of computing a deterministic score like Expected Improvement,
    we sample a value from the posterior distribution for each candidate point
    and select the one with the lowest sampled value (by returning negative
    samples as scores, since higher score is better).
    
    This approach naturally balances exploration and exploitation:
    - Points with low uncertainty (small sigma) will have samples close to their mean.
    - Points with high uncertainty (large sigma) will have samples that vary widely,
      giving them a chance to be selected if their sample happens to be very low.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and handle potential numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Sample from the posterior: y ~ N(mu, sigma^2)
    # We generate standard normal random numbers and scale/shift them
    z = np.random.standard_normal(mu.shape)
    samples = mu + sigma_safe * z
    
    # We want to minimize the function.
    # The evaluator picks the point with the HIGHEST score.
    # So we return the negative of the sampled value.
    scores = -samples
    
    # Ensure finiteness
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
