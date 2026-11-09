"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_DEADLINE = 330.0
_START_TIME = None

def _get_elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _check_deadline():
    return _get_elapsed() < _DEADLINE - 5.0  # 5s safety margin

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Weierstrass 1D.
    
    Uses Adaptive Upper Confidence Bound (UCB) for minimization.
    Score = -mu + beta * sigma
    
    The beta parameter is adapted based on f_best to balance exploration and exploitation:
    - Early in optimization (high f_best), beta is high to encourage exploration of unexplored regions.
    - Late in optimization (low f_best), beta decreases to favor exploitation and convergence.
    
    This structural change from the previous hybrid EI/PI approach leverages the robustness 
    of UCB for rugged, multimodal landscapes like Weierstrass, while the adaptive beta 
    refines the search efficiency as the global minimum is approached.
    """
    # Check time budget
    if not _check_deadline():
        # Fallback to simple exploitation if we're close to deadline
        return -mu

    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Adaptive beta:
    # f_best is in objective units. Weierstrass min is 0.
    # Typical initial f_best might be around 0.5-1.0.
    # We map f_best to a beta range.
    # If f_best is high (e.g., 1.0), we want high exploration (beta ~ 4.0).
    # If f_best is low (e.g., 0.0), we want low exploration (beta ~ 1.0).
    
    # Clip f_best to a reasonable range to prevent extreme beta values
    # If f_best is negative (shouldn't happen if min is 0, but safe), clip to 0.
    fb = np.clip(f_best, 0.0, 1.0)
    
    # Linear adaptation
    beta = 1.0 + 3.0 * fb
    
    # UCB score for minimization:
    # We maximize the score.
    # -mu favors low predicted values.
    # beta * sigma favors high uncertainty.
    score = -mu + beta * sigma_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
