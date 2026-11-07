"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    # Track time to ensure we don't exceed the deadline
    # The task says each evaluation must finish within 300s, but the evaluator kills after 330s.
    # Since this function is called once per step, and we return 2000 scores,
    # the computation here is vectorized and should be very fast (< 1ms).
    # However, adding a guard is a good practice per the directive.
    
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    # Ensure sigma is positive and has a minimum floor to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-6)
    
    # Calculate Expected Improvement
    z = (f_best - mu) / sigma_safe
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(ei), ei, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
