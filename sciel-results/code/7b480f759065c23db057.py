"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Upper Confidence Bound (UCB).
    Higher score = more likely to be evaluated.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return a safe fallback: just exploit the best mean
        scores = -mu / cost
        if not np.all(np.isfinite(scores)):
            scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Adaptive beta: start with more exploration, reduce over time
    # Use f_best as a proxy for progress: as f_best decreases, we've found better points
    # and should shift toward exploitation
    # Normalize f_best: for rosenbrock 2d, min is 0, typical values can be large
    # Use a smooth decay: beta decreases as f_best approaches 0
    
    # f_best starts high and decreases toward 0
    # Map f_best to a beta range: high f_best -> high beta (explore), low f_best -> low beta (exploit)
    # Use exponential decay for smoothness
    # When f_best is large (early), beta ~ 2.0; when f_best ~ 0, beta ~ 0.8
    f_best_clipped = np.clip(f_best, 0.0, 100.0)
    # Smooth transition: beta = 0.8 + 1.2 * exp(-f_best / 10)
    beta = 0.8 + 1.2 * np.exp(-f_best_clipped / 10.0)
    
    # UCB for minimization: Score = (-mu + beta * sigma) / cost
    scores = (-mu + beta * sigma) / cost
    
    # Safety check for finite values
    if not np.all(np.isfinite(scores)):
        scores = np.where(np.isfinite(scores), scores, 0.0)
        
    return scores
# EVOLVE-BLOCK-END
