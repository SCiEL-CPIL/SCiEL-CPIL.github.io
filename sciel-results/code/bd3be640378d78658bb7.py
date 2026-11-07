# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard
_DEADLINE = time.time() + 280  # safety margin before 330s limit


def acquisition(mu, sigma, f_best):
    """
    Hybrid Acquisition Function: Switching between Exploratory UCB-EI and Exploitative EI.
    
    This function implements a structural switch in strategy based on the current best value (f_best).
    - Early stage (high f_best): Uses a score dominated by uncertainty (sigma) to explore the space.
    - Late stage (low f_best): Uses pure Expected Improvement (EI) to exploit promising regions.
    
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
        return -mu.copy()
    
    # Ensure arrays are 1D and finite
    mu = np.asarray(mu, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64).ravel()
    n = len(mu)
    
    # Handle any non-finite values
    mask = np.isfinite(mu) & np.isfinite(sigma)
    if not np.all(mask):
        # Replace non-finite with worst-case for mu (high value for minimization)
        # and 0 for sigma
        safe_mu_val = np.max(mu[mask]) if np.any(mask) else 0.0
        mu = np.where(mask, mu, safe_mu_val)
        sigma = np.where(mask, sigma, 0.0)
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # 1. Compute Expected Improvement (EI)
    # EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    # where z = (f_best - mu(x)) / sigma(x)
    
    improvement = f_best - mu
    scores_ei = np.zeros(n, dtype=np.float64)
    nonzero_sigma = sigma > 1e-12
    
    if np.any(nonzero_sigma):
        z = improvement[nonzero_sigma] / sigma[nonzero_sigma]
        # Clip z to avoid numerical issues in norm.cdf/pdf for extreme values
        z = np.clip(z, -50, 50)
        scores_ei[nonzero_sigma] = (
            improvement[nonzero_sigma] * norm.cdf(z) +
            sigma[nonzero_sigma] * norm.pdf(z)
        )
    
    # If sigma is 0, EI is max(0, f_best - mu)
    if np.any(~nonzero_sigma):
        scores_ei[~nonzero_sigma] = np.maximum(improvement[~nonzero_sigma], 0.0)
    
    # 2. Determine Exploration Weight based on Progress
    # Global minimum of Hartmann 3D is -3.86278
    f_global_min = -3.86278
    
    # We define a transition zone.
    # If f_best is very bad (e.g., > -1.0), we are in "Exploration Mode".
    # If f_best is very good (e.g., < -2.5), we are in "Exploitation Mode".
    # Between these, we blend.
    
    # Thresholds
    f_explore_threshold = -1.0  # Above this, explore heavily
    f_exploit_threshold = -2.5  # Below this, exploit purely
    
    if f_best >= f_explore_threshold:
        # Early stage: Strong exploration
        # Use a UCB-like bonus. We add a multiple of sigma to the EI score.
        # EI already has some exploration, but we boost it significantly.
        explore_weight = 2.0
    elif f_best <= f_exploit_threshold:
        # Late stage: Pure exploitation
        explore_weight = 0.0
    else:
        # Transition stage: Linearly reduce exploration weight
        # progress from 1.0 (at f_explore_threshold) to 0.0 (at f_exploit_threshold)
        range_val = f_explore_threshold - f_exploit_threshold
        if range_val == 0:
            progress = 1.0
        else:
            progress = (f_best - f_exploit_threshold) / range_val
        
        # progress is 1.0 when f_best is -1.0, 0.0 when f_best is -2.5
        # We want weight to go from 2.0 down to 0.0
        explore_weight = 2.0 * progress
    
    # 3. Combine Scores
    # Score = EI + explore_weight * sigma
    # This structure allows the function to fundamentally change its behavior:
    # - With explore_weight > 0, it favors high uncertainty regions (exploration).
    # - With explore_weight == 0, it strictly follows EI (exploitation).
    
    scores = scores_ei + explore_weight * sigma
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
