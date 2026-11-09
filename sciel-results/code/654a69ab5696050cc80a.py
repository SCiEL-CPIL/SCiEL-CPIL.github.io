"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Ensure sigma is strictly positive to avoid division by zero
    # Use a slightly larger floor to handle numerical noise in GP predictions
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate the standardized improvement z-score
    # We scale the z-score by a factor of 0.5 (equivalent to dividing by 2*sigma)
    # to reduce the exploitation bias of standard POI.
    # This creates a flatter landscape, encouraging exploration of high-variance
    # regions which may contain the global minimum in multimodal functions like Branin.
    z = (f_best - mu) / (sigma_safe * 2.0)
    
    # Probability of Improvement: P(f(x) < f_best) = Phi(z)
    # Higher probability means higher score
    scores = norm.cdf(z)
    
    # Ensure scores are finite and within [0, 1]
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
