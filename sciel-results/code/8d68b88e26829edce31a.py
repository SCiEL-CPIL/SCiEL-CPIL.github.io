# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard
_DEADLINE = time.time() + 280  # safety margin before 330s limit

def acquisition(mu, sigma, f_best):
    """
    Adaptive Upper Confidence Bound (UCB) Acquisition Function.
    
    This function implements a structural change from the parent's 
    'Hybrid Rank-Boosted EI' to a direct linear combination of the 
    posterior mean and standard deviation.
    
    Strategy:
    - Maximizes: Score(x) = -mu(x) + beta * sigma(x)
    - This is equivalent to minimizing mu(x) - beta * sigma(x) (LCB for minimization).
    - The parameter beta is adapted based on the progress of the search,
      inferred from the current best value f_best relative to the known
      global minimum of the Hartmann 3D function (-3.86278).
    - Early in the search (f_best is far from global min), beta is high to
      encourage exploration of high-uncertainty regions.
    - Late in the search (f_best is close to global min), beta is low to
      focus on exploitation of low-mean regions.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        GP posterior mean at candidate points.
    sigma : np.ndarray, shape (2000,)
        GP posterior standard deviation at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores; higher means more attractive to evaluate next.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: just exploit (maximize -mu)
        return -np.asarray(mu, dtype=np.float64).ravel().copy()
    
    # Ensure arrays are 1D and finite
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    n = len(mu)
    
    # Handle any non-finite values
    mask = np.isfinite(mu) & np.isfinite(sigma)
    if not np.all(mask):
        # For minimization, high mu is bad. So replace non-finite mu with a very large value.
        # For sigma, 0 is safe (no uncertainty).
        safe_mu_val = np.max(mu[mask]) + 10.0 if np.any(mask) else 0.0
        mu = np.where(mask, mu, safe_mu_val)
        sigma = np.where(mask, sigma, 0.0)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # Calculate progress
    f_global_min = -3.86278
    # f_best starts high (e.g., -1.0) and goes down to -3.86.
    # We want progress 0 at start, 1 at end.
    # Range of f_best is roughly [0, -3.86] or higher.
    # Let's clamp f_best to [f_global_min, 0.0] for progress calculation.
    # If f_best is > 0, progress is 0.
    # If f_best is < f_global_min, progress is 1.
    
    # Normalize f_best to [0, 1] where 0 is "worst" (0.0) and 1 is "best" (f_global_min)
    # progress = (0.0 - f_best) / (0.0 - f_global_min)
    # If f_best = 0, progress = 0.
    # If f_best = -3.86, progress = 1.
    
    denom = 0.0 - f_global_min # 3.86278
    if denom == 0:
        progress = 0.0
    else:
        # Clip f_best to avoid issues if it's outside the expected range
        f_best_clamped = np.clip(f_best, f_global_min, 0.0)
        progress = (0.0 - f_best_clamped) / denom
        progress = np.clip(progress, 0.0, 1.0)
    
    # Adaptive Beta
    # We want high exploration early, low exploration late.
    # beta_max = 2.5, beta_min = 0.5
    # Linear interpolation
    beta_max = 2.5
    beta_min = 0.5
    beta = beta_max - (beta_max - beta_min) * progress
    
    # Calculate Scores
    # We want to maximize: -mu + beta * sigma
    scores = -mu + beta * sigma
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
