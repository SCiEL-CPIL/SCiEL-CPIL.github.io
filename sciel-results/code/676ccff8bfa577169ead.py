import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Expected Improvement (EI) for minimization
    # This corresponds to the greedy maximization of Hypervolume Improvement
    # with reference point f_best.
    
    # Handle numerical stability for division by sigma
    eps = 1e-8
    safe_sigma = np.maximum(sigma, eps)
    
    # Z-score for improvement
    Z = (f_best - mu) / safe_sigma
    
    # Calculate EI
    # E[I] = (f_best - mu) * Phi(Z) + sigma * phi(Z)
    cdf = norm.cdf(Z)
    pdf = norm.pdf(Z)
    
    scores = (f_best - mu) * cdf + safe_sigma * pdf
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
