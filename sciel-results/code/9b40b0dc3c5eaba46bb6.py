import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    Uses a blend of Expected Improvement (EI) and a diversity/repulsion term
    to encourage exploration of unexplored regions while exploiting promising ones.
    """
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma
    
    # Standard EI for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute a repulsion/diversity term based on uncertainty
    # Points with high sigma are "unexplored" - we want to encourage them
    # but not too aggressively. Use a scaled version of sigma.
    # Normalize sigma to [0, 1] range for blending
    sigma_min = np.min(sigma)
    sigma_max = np.max(sigma)
    sigma_range = sigma_max - sigma_min + 1e-9
    sigma_norm = (sigma - sigma_min) / sigma_range
    
    # Blend EI with a diversity term
    # The diversity term rewards high uncertainty (unexplored regions)
    # We use a weighted combination: primarily EI, with a small boost for uncertainty
    alpha = 0.3  # Weight for diversity term
    
    scores = ei + alpha * sigma_norm * (np.max(ei) - np.min(ei) + 1e-9)
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return scores
