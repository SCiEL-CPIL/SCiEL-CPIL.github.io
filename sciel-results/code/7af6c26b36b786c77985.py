"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: blend of Expected Improvement and a normalized exploitation term.
    
    The exploitation term (-mu) directly guides search toward regions with low predicted
    mean, which is crucial for multi-modal landscapes like Hartmann-3D where EI can be
    misleading in unexplored regions.
    
    Returns 2000 finite scores; higher score means evaluate next.
    """
    # Deadline guard: track elapsed time to avoid timeout
    start_time = time.time()
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute exploitation term: negative mean (lower mean is better)
    # Normalize mu to [0, 1] range for blending
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = mu_max - mu_min
    if mu_range < 1e-9:
        mu_norm = np.zeros_like(mu)
    else:
        mu_norm = (mu - mu_min) / mu_range  # 0 at best (lowest), 1 at worst (highest)
    
    # Exploitation score: lower mu gives higher score
    exploit = 1.0 - mu_norm
    
    # Blend: heavily weight exploitation, add EI as secondary signal
    # The blend addresses the weakness of EI in multi-modal landscapes
    alpha = 0.7  # weight for exploitation
    beta = 0.3   # weight for EI
    
    # Normalize EI to [0, 1] for consistent blending
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = ei_max - ei_min
    if ei_range < 1e-9:
        ei_norm = np.zeros_like(ei)
    else:
        ei_norm = (ei - ei_min) / ei_range
    
    # Combined score
    score = alpha * exploit + beta * ei_norm
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Safety margin check
    elapsed = time.time() - start_time
    if elapsed > 280:  # 20 second safety margin before 300s limit
        return score
    
    return score
# EVOLVE-BLOCK-END
