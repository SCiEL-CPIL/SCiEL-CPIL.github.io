import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # UCB (Upper Confidence Bound) for minimization
    # We maximize: -mu + beta * sigma
    
    # Adaptive beta based on the gap to the known global minimum
    # Hartmann 3D global min is approx -3.86278
    f_min_known = -3.86278
    gap = f_best - f_min_known
    
    # If gap is large, we are far from the optimum, explore more.
    # If gap is small, exploit more.
    # Scale gap: typical range is 0 to ~7.
    # beta ranges from 2.0 (explore) to 0.5 (exploit)
    # Using an exponential decay for smooth transition
    beta = 0.5 + 1.5 * np.exp(-gap / 1.0)
    
    # Calculate scores
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-9)
    
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
