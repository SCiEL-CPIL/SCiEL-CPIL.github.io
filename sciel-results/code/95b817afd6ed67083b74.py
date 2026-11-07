import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    Samples from the GP posterior and returns the negative sample value
    so that the point with the lowest sampled value gets the highest score.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and handle numerical issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Generate standard normal random variables
    # Using a fixed seed inside the function is not allowed as it would make it deterministic 
    # and potentially stuck. We rely on the global RNG state or just np.random.
    # However, in BO loops, reproducibility isn't strictly required per step, 
    # but we should ensure we don't crash.
    z = np.random.standard_normal(mu.shape)
    
    # Sample function values
    y_samples = mu + sigma_safe * z
    
    # We want the evaluator to pick the point with the lowest objective.
    # The evaluator picks the highest score.
    # So we return the negative of the sample.
    score = -y_samples
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
