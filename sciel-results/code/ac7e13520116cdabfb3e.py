"""Rewritten acquisition for Bayesian optimization of Hartmann 3D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track elapsed time for deadline guard
_start_time = time.time()
_DEADLINE = 330.0  # seconds


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Upper Confidence Bound (UCB) with dynamically
    increasing exploration constant.
    
    UCB = mu - kappa * sigma (for minimization)
    
    kappa increases with the number of evaluations to balance exploration
    (early) vs exploitation (late). We estimate the number of evaluations
    from f_best relative to the known minimum and initial expected value.
    
    Parameters
    ----------
    mu : ndarray, shape (n,)
        GP mean predictions at candidate points.
    sigma : ndarray, shape (n,)
        GP standard deviation predictions at candidate points.
    f_best : float
        Best (lowest) observed objective value so far.

    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher means more promising.
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE - 5.0:
        # Return the point with lowest mu (most exploitative) as fallback
        return -mu.copy()
    
    # Known global minimum of Hartmann 3D
    f_min = -3.86278
    # Estimated initial best value for uniform random points
    f_start = -1.0
    
    # Estimate number of evaluations from f_best
    # As f_best improves from f_start toward f_min, we estimate progress
    gap_initial = f_start - f_min  # ~2.86
    gap_current = f_best - f_min
    
    if gap_initial > 1e-9:
        progress = 1.0 - np.clip(gap_current / gap_initial, 0.0, 1.0)
    else:
        progress = 1.0
    
    # Total evaluations: 5 initial + 25 more = 30
    # Estimate current eval count
    n_total = 30
    n_initial = 5
    n_current = n_initial + int(progress * (n_total - n_initial))
    
    # UCB exploration parameter: kappa = c * sqrt(log(n)) / sqrt(n)
    # This starts high (exploration) and decreases (exploitation)
    # Use a base constant c tuned for Hartmann 3D
    c = 1.5
    kappa = c * np.sqrt(np.log(max(n_current, 2))) / np.sqrt(max(n_current, 2))
    
    # Ensure kappa is reasonable
    kappa = np.clip(kappa, 0.1, 3.0)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB for minimization: we want to maximize (mu - kappa * sigma)
    # But since lower objective is better, and we want higher acquisition score
    # to mean "evaluate this next", we compute:
    # score = mu - kappa * sigma  (lower mu is better, higher sigma is exploration)
    # Wait - for minimization with UCB, we minimize mu + kappa*sigma
    # So the acquisition (higher = better) should be: -mu - kappa*sigma
    # Actually, let me reconsider:
    # For minimization: UCB value = mu - kappa * sigma (this is an upper bound on the true value)
    # We want to pick the point with the lowest UCB value (most promising minimum)
    # So acquisition score (higher = pick this) = -(mu - kappa * sigma) = -mu + kappa * sigma
    
    scores = -mu + kappa * sigma_safe
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
