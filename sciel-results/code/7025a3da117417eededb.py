import numpy as np
from scipy.stats import norm
import time

# Global state to track optimization progress
_call_count = 0

def acquisition(mu, sigma, f_best):
    global _call_count
    
    # 1. Increment step counter
    _call_count += 1
    
    # 2. Determine step progress
    # Total evaluations = 30. Initial 5 are random. So 25 acquisition calls.
    max_steps = 25
    if _call_count > max_steps:
        _call_count = max_steps
        
    progress = _call_count / max_steps

    # 3. Define exploration floor for sigma
    # We want to ensure early steps explore broadly.
    # Ackley 2D has values roughly in [0, ~7].
    # A sigma floor of 0.5 is significant.
    # Decay from 0.5 to 0.01 exponentially or linearly?
    # Linear decay is simple and effective.
    sigma_start = 0.5
    sigma_end = 0.01
    min_sigma = sigma_start + (sigma_end - sigma_start) * progress
    
    # 4. Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # z = (f_best - mu) / sigma
    
    # Ensure sigma is at least min_sigma
    sigma_safe = np.maximum(sigma, min_sigma)
    
    # Calculate z
    # Avoid division by zero, though sigma_safe >= min_sigma > 0
    z = (f_best - mu) / sigma_safe
    
    # Phi(z) and phi(z)
    # norm.cdf(z) and norm.pdf(z)
    # For numerical stability, we can compute directly.
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # 5. Safety checks
    # Ensure finite scores
    scores = np.where(np.isfinite(ei), ei, -np.inf)
    
    # If all scores are -inf (shouldn't happen), return zeros
    if np.all(scores == -np.inf):
        return np.zeros_like(mu)
        
    return scores
