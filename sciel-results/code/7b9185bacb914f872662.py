"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global variables for deadline guard
_start_time = None
_deadline = 300.0
_safety_margin = 5.0

def _check_deadline():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    if time.time() > _start_time + _deadline - _safety_margin:
        return True
    return False

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    # Ensure sigma is positive for division
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score for Expected Improvement
    # mu, sigma, f_best are arrays
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (minimization)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z)
    phi_z_f_best = norm.cdf(z)
    
    ei = (f_best - mu) * phi_z_f_best + sigma_safe * phi_z
    
    # Cost-aware score
    # cost is array of shape (2000,)
    # We want to maximize score
    # Score = EI / Cost
    # Add a bounded exploration bonus to encourage visiting uncertain regions
    # The term sigma/(sigma+0.1) is bounded [0, 1) and increases with sigma
    exploration_bonus = sigma_safe / (sigma_safe + 0.1)
    
    cost_safe = np.maximum(cost, 1e-8)
    
    # Combine EI and exploration bonus
    # Small weight for exploration to not overpower exploitation
    scores = (ei + 0.1 * exploration_bonus) / cost_safe
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
