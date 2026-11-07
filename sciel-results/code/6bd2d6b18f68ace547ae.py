"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_DEADLINE = None
_START_TIME = None

def _ensure_deadline():
    global _DEADLINE, _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        _DEADLINE = _START_TIME + 320.0  # 320s safety margin within 330s limit
    return _DEADLINE

def acquisition(mu, sigma, f_best):
    deadline = _ensure_deadline()
    
    # Check if we're close to deadline
    if time.time() > deadline - 2.0:
        # Return a simple EI to avoid timeout
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Hybrid acquisition: EI + small exploration term
    # EI component
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Exploration term: small bonus for high uncertainty
    # Scale sigma relative to the range of observed values
    sigma_max = np.max(sigma) if len(sigma) > 0 else 1.0
    sigma_min = np.min(sigma) if len(sigma) > 0 else 0.0
    sigma_range = sigma_max - sigma_min + 1e-9
    
    # Normalized sigma in [0, 1]
    sigma_norm = (sigma - sigma_min) / sigma_range
    
    # Small exploration bonus (tuned to be modest)
    exploration_weight = 0.1
    exploration = exploration_weight * sigma_norm
    
    # Combined score
    scores = ei + exploration
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
