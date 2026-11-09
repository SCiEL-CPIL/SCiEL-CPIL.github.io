# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    # Deadline guard: ensure we return within time budget
    t_start = time.time()
    
    # Ensure sigma is positive and has valid shape
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standard EI: (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    diff = f_best - mu
    z = diff / sigma_safe
    
    # Use scipy.stats.norm for numerical stability
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = diff * cdf_z + sigma_safe * pdf_z
    
    # Bounded exploration bonus: encourages exploration in uncertain regions
    # while being damped when the point is far from f_best
    bonus = 0.05 * sigma_safe / (np.abs(diff) + sigma_safe + 1e-6)
    
    score = ei + bonus
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -1e10)
    
    # Check if we're close to the deadline; if so, just return
    elapsed = time.time() - t_start
    if elapsed > 280:  # 300s limit with safety margin
        return score
    
    return score
# EVOLVE-BLOCK-END
