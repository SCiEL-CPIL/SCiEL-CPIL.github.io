import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is strictly positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Structural Change: Use Lower Confidence Bound (LCB) directly
    # LCB(x) = mu(x) - beta * sigma(x)
    # We want to minimize LCB, so we maximize Score = -LCB = beta * sigma - mu
    # This avoids the normalization issues of the previous blend.
    
    # Adaptive beta based on the current best value's scale?
    # No, let's use a fixed beta for simplicity and robustness.
    # A beta of 2.5 is a reasonable default for 1D BO with Matern-5/2.
    beta = 2.5
    
    score = beta * sigma - mu
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
