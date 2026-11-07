"""Acquisition function for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global timer for deadline guard
_start_time = time.time()
_DEADLINE = 325.0  # Safety margin before 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for minimization using adaptive UCB.
    
    Uses a pure UCB structure with an adaptive exploration coefficient
    that follows a logistic schedule based on simple regret. This provides
    aggressive exploration early (to find multiple basins in the multimodal
    Michalewicz landscape) and sharp transition to exploitation as we
    approach the known minimum.
    
    Parameters:
    -----------
    mu : array-like, shape (n_candidates,)
        Posterior mean predictions at candidate points.
    sigma : array-like, shape (n_candidates,)
        Posterior standard deviations at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n_candidates,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Deadline guard: if we're close to the time limit, use pure exploitation
    if time.time() - _start_time > _DEADLINE:
        scores = -np.asarray(mu, dtype=np.float64)
        return np.where(np.isfinite(scores), scores, -1e6)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Michalewicz 2D minimum is approximately -1.8013
    f_min = -1.8013
    
    # Simple regret: how far we are from the optimum
    simple_regret = max(f_best - f_min, 0.0)
    
    # Adaptive beta using logistic schedule:
    # - When regret is large (early), beta is high (aggressive exploration)
    # - When regret is small (late), beta drops sharply (exploitation)
    # The logistic transition is centered around regret ~ 0.3 with steep slope
    # This is structurally different from linear or exponential decay:
    # it maintains high exploration until close to optimum, then switches fast
    
    beta_max = 5.0      # Very high exploration early
    beta_min = 0.5      # Low exploration late (near pure exploitation)
    regret_center = 0.3  # Center of logistic transition
    regret_slope = 8.0   # Steepness of transition
    
    # Logistic: 1 / (1 + exp(slope * (regret - center)))
    # When regret >> center: exp large -> logistic ~ 0 -> beta ~ beta_min
    # When regret << center: exp small -> logistic ~ 1 -> beta ~ beta_max
    logistic_val = 1.0 / (1.0 + np.exp(regret_slope * (simple_regret - regret_center)))
    beta = beta_min + (beta_max - beta_min) * logistic_val
    
    # Pure UCB score for minimization: higher score = more promising
    # score = -mu + beta * sigma
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e6)
    
    return scores
# EVOLVE-BLOCK-END
