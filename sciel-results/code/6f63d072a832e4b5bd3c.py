"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track elapsed time and enforce the 300s deadline
_start_time = time.time()
_DEADLINE = 300.0
_SAFETY_MARGIN = 5.0  # seconds before the hard limit

def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores using a deterministic Repulsion-Gravity heuristic.
    
    Score = (f_best - mu) * sigma
    
    This heuristic balances:
    1. Gravity: Attraction to points where the mean prediction (mu) is lower than f_best.
    2. Repulsion/Exploration: Weighting by the local standard deviation (sigma).
       High sigma indicates high uncertainty (unexplored regions), which acts as a repulsive
       force from the currently "explored" low-variance areas.
    
    This is structurally different from EI because it does not model the probability of
    improvement via the normal CDF, but rather uses a simple multiplicative interaction
    between the potential improvement and the uncertainty. This can be more aggressive
    in exploration when uncertainty is high, even if the mean is not yet promising,
    and it avoids the complex tail behavior of EI.
    
    Parameters:
    -----------
    mu : array-like, shape (n,)
        Mean of the GP posterior at candidate points.
    sigma : array-like, shape (n,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores for each candidate point.
    """
    # Deadline guard: if we are close to the limit, return a safe, deterministic result
    # that effectively selects the point with the lowest mean (exploitation) to avoid
    # wasting time on complex calculations or random choices that might fail.
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        # Fallback: exploit the best mean
        mu = np.asarray(mu, dtype=np.float64)
        scores = -mu
        scores[~np.isfinite(scores)] = 0.0
        return scores

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Clip sigma to avoid numerical issues with zero variance
    sigma_clipped = np.clip(sigma, 1e-9, None)
    
    # Repulsion-Gravity Score:
    # We want to maximize the score.
    # (f_best - mu) is positive when mu < f_best (improvement potential).
    # sigma is the uncertainty.
    # Multiplying them creates a score that is high when both improvement potential
    # and uncertainty are high.
    
    # Note: If f_best - mu is negative (mu > f_best), the score becomes negative.
    # This naturally penalizes points that are predicted to be worse than the current best,
    # even if they have high uncertainty, unless the uncertainty is extremely high 
    # (which might still be worth exploring if the mean is only slightly worse).
    # However, for a pure "Repulsion-Gravity" that encourages exploration of unknown areas,
    # we might want to ensure that high uncertainty always has a chance.
    # But the standard form (f_best - mu) * sigma is a common heuristic (similar to 
    # a simplified UCB or EI variant).
    
    # Let's use the standard form:
    scores = (f_best - mu) * sigma_clipped
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
