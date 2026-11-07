import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI).
    Returns scores of shape (2000,) where higher means evaluate next.
    """
    # Convert to numpy arrays for safety
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and add epsilon to avoid division by zero
    sigma = np.maximum(sigma, 1e-12)
    
    # Small exploration parameter xi to encourage slight exploration
    # For 1D sphere, a small xi is usually sufficient and better than 0
    # because pure EI can get stuck in local minima if the GP is overconfident
    # due to the small length scale and noise.
    xi = 0.05
    
    # Calculate z-score
    z = (f_best - xi - mu) / sigma
    
    # Calculate Expected Improvement
    # EI = (f_best - xi - mu) * Phi(z) + sigma * phi(z)
    # For minimization, we want to maximize the probability of finding a lower value.
    # The standard EI formula assumes minimization.
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    scores = (f_best - xi - mu) * cdf + sigma * pdf
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
