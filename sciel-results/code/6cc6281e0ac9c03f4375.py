import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition for minimization.
    
    Samples from the GP posterior and returns negative samples as scores.
    Lower sampled values mean higher scores (since we want to minimize).
    
    Args:
        mu: GP posterior mean, shape (N,)
        sigma: GP posterior std, shape (N,)
        f_best: Best observed value so far (lower is better)
        
    Returns:
        Scores for each candidate, shape (N,). Higher scores preferred.
    """
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Sample from the posterior: theta ~ N(mu, sigma^2)
    # Use a fixed random seed per call? No, we want different samples each time.
    # But we need reproducibility within a single call for the 2000 candidates.
    # Actually, we just need one sample per candidate.
    
    # Generate random normal samples
    samples = mu + sigma_safe * np.random.randn(len(mu))
    
    # For minimization, we want to pick the candidate with the LOWEST sample
    # So our score should be HIGH when the sample is LOW
    # Score = -sample (negate so lower function values get higher scores)
    scores = -samples
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
