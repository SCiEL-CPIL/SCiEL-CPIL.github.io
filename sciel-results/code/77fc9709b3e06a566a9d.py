import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() - _start_time > 320:
        return -mu

    # Ensure sigma is positive and finite
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Expected Improvement (EI) for minimization
    # We want to maximize the expected improvement over f_best.
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    # Add a small exploration parameter xi (jitter)
    # This encourages the algorithm to seek points that are better than f_best by at least xi
    # A small xi helps escape local optima in the early stages
    xi = 0.01
    
    z = (f_best - xi - mu) / sigma_safe
    
    # Phi(z) is the CDF of the standard normal
    # phi(z) is the PDF of the standard normal
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # EI score
    ei_scores = (f_best - xi - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure finite scores
    ei_scores = np.where(np.isfinite(ei_scores), ei_scores, -1e10)
    
    return ei_scores
