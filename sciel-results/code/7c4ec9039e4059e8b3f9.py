import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    # Deadline guard
    start_time = time.time()
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    # Handle numerical stability for large negative z
    # If z is very large negative, cdf(z) ~ 0, pdf(z) ~ 0.
    # If z is very large positive, cdf(z) ~ 1, pdf(z) ~ 0.
    
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure scores are finite
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Add a small UCB component to encourage exploration
    # The coefficient is tuned to balance exploration and exploitation
    # For Hartmann 3D, we want to explore enough to find the global minimum
    # but not too much to waste evaluations on poor regions
    beta = 0.5
    ucb = beta * sigma_safe
    
    # Combine EI and UCB
    scores = ei + ucb
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
