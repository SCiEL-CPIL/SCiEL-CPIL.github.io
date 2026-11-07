"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timer for deadline guard (though acquisition is fast)
_start_time = time.time()
_DEADLINE = 300.0  # Safety margin from 330s kill

def acquisition(mu, sigma, f_best):
    # Check deadline
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Return a valid array of zeros or something finite to avoid crash
        # But the evaluator kills us anyway. Just return fast.
        return np.zeros_like(mu)

    # Handle numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate standard EI
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Entropy maximization proxy: log(sigma^2)
    # This targets regions of highest posterior uncertainty
    # Normalizing log_sigma to have zero mean and unit variance for consistent scaling
    log_sigma = np.log(sigma_safe)
    log_sigma_mean = np.mean(log_sigma)
    log_sigma_std = np.std(log_sigma)
    if log_sigma_std > 1e-9:
        log_sigma_norm = (log_sigma - log_sigma_mean) / log_sigma_std
    else:
        log_sigma_norm = np.zeros_like(log_sigma)
    
    # Combine EI and normalized log-sigma
    # Weight 0.5 for exploration via entropy
    # This balances exploitation (EI) with exploration (uncertainty)
    scores = ei + 0.5 * log_sigma_norm
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
