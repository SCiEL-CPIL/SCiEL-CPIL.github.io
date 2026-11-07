"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit

    # Handle edge cases
    if len(mu) == 0:
        return np.array([])
    
    # Ensure finite values
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    sigma = np.maximum(sigma, 1e-9)
    
    n = len(mu)
    
    # Probability of Improvement (PoI)
    # We want to maximize the probability that f(x) < f_best
    # Given f(x) ~ N(mu, sigma^2), P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    # Phi is the CDF of the standard normal distribution.
    
    # To avoid division by zero or very small numbers, we already ensured sigma >= 1e-9.
    # However, if sigma is extremely small and mu > f_best, the z-score will be large negative -> Phi ~ 0.
    # If sigma is extremely small and mu < f_best, the z-score will be large positive -> Phi ~ 1.
    
    # Calculate z-scores
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Clip z to a reasonable range to avoid numerical issues in norm.cdf, 
    # though norm.cdf handles large values well (returns 0.0 or 1.0).
    # For very large z, PoI is effectively 1.0.
    # For very small z, PoI is effectively 0.0.
    
    # Compute PoI
    poi_scores = norm.cdf(z)
    
    # Ensure all scores are finite
    poi_scores = np.where(np.isfinite(poi_scores), poi_scores, 0.0)
    
    # Optional: Add a tiny bit of exploration noise to break ties or encourage diversity
    # if the GP is overconfident (sigma ~ 0) in unexplored areas? 
    # Actually, PoI naturally handles this: if sigma is small, it's either 0 or 1.
    # If sigma is large, it's a smooth gradient.
    
    # Return the scores. Higher score = higher probability of improvement = evaluate next.
    return poi_scores
# EVOLVE-BLOCK-END
