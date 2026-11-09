import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function using sampled posterior paths.
    
    Instead of computing the probability of improvement (which can be too 
    greedy or too uniform depending on the prior), we sample a single 
    realization of the function value for each candidate point from the 
    posterior distribution N(mu, sigma^2).
    
    The point with the lowest sampled value is selected for the next evaluation.
    This approach naturally balances exploration (high sigma leads to large 
    variance in samples, potentially uncovering new minima) and exploitation 
    (low sigma leads to samples close to mu, refining known good areas).
    
    Args:
        mu: array of shape (N,) with posterior means.
        sigma: array of shape (N,) with posterior standard deviations.
        f_best: float, the best value observed so far (unused in pure TS, 
                but kept for signature compatibility).
        
    Returns:
        array of shape (N,) with scores. Higher scores indicate points 
        that are more likely to be the next best evaluation.
        Note: Since we are minimizing, we return -sampled_value so that 
        higher scores correspond to lower (better) function values.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive to avoid division by zero or invalid samples
    sigma = np.maximum(sigma, 1e-8)
    
    # Generate a standard normal random sample for each candidate
    # Shape is (N,)
    z = np.random.standard_normal(len(mu))
    
    # Sample from the posterior: f_i ~ N(mu_i, sigma_i^2)
    # f_sample = mu + sigma * z
    f_samples = mu + sigma * z
    
    # We want to minimize f. The evaluator picks the candidate with the 
    # HIGHEST score. 
    # Therefore, we should return scores that are higher for lower f_samples.
    # A simple and effective transformation is the negative of the sample.
    # score = -f_sample
    # This ensures that the point with the lowest sampled function value 
    # gets the highest score.
    
    scores = -f_samples
    
    # Ensure finite outputs. 
    # If mu or sigma contains inf/nan, the score will be non-finite.
    # The contract requires finite scores.
    # We replace non-finite values with a very low score (effectively -inf) 
    # so they are never selected.
    scores = np.where(np.isfinite(scores), scores, -1e30)
    
    return scores
