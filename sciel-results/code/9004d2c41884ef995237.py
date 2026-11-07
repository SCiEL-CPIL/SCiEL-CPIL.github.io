"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = _T0 + 320.0

def acquisition(mu, sigma, f_best):
    # UCB acquisition for minimization
    # We want to minimize mu + kappa * sigma
    # The score returned should be higher for better candidates
    # So we return -(mu + kappa * sigma)
    
    # Check deadline (though computation is trivial for 2000 points)
    if time.time() > _DEADLINE - 5.0:
         # Fallback to simple exploitation if time is up (unlikely)
         return -mu

    kappa = 2.0
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-8)
    
    scores = -(mu + kappa * sigma)
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
