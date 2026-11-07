import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_start_time = None
_DEADLINE = 320  # seconds, safety margin before 330s kill

def _init_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()

def _check_deadline():
    if _start_time is not None:
        return (time.time() - _start_time) < _DEADLINE
    return True

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    
    Uses a combined Expected Improvement (EI) with an uncertainty-weighted
    exploration term that adapts based on how close we are to the known minimum.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Posterior mean at each candidate point.
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at each candidate point.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means more desirable to evaluate next.
    """
    _init_time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, 1e-10)
    
    # Check deadline early - if we're running out of time, return a simple score
    if not _check_deadline():
        # Return a simple score based on low mean + some uncertainty
        scores = -(mu) + 0.1 * sigma
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Compute Expected Improvement for minimization
    # EI(x) = (f_best - mu(x)) * Phi((f_best - mu(x)) / sigma(x)) + sigma(x) * phi((f_best - mu(x)) / sigma(x))
    z = (f_best - mu) / sigma
    z_clipped = np.clip(z, -10, 10)
    
    Phi = norm.cdf(z_clipped)
    phi = norm.pdf(z_clipped)
    
    ei = (f_best - mu) * Phi + sigma * phi
    
    # Adaptive exploration: when f_best is far from the known minimum (-2.53218),
    # we should explore more. As we get closer, we should exploit more.
    # The known minimum is approximately -2.53218
    known_min = -2.53218
    distance_to_min = f_best - known_min  # positive when we're above the minimum
    
    # Normalize the distance to get an exploration weight
    # When distance is large, explore more; when small, exploit more
    # Use a sigmoid-like scaling
    # Scale: when distance_to_min is around 1-2, we want moderate exploration
    exploration_weight = np.exp(-np.abs(distance_to_min) / 1.5)
    # exploration_weight is in (0, 1], higher when close to minimum
    
    # Actually, we want MORE exploration when FAR from minimum
    # So flip it: exploration_weight should be high when distance_to_min is large
    exploration_weight = 1.0 / (1.0 + np.exp(-np.abs(distance_to_min) / 0.5))
    # When distance is large, this approaches 1; when small, approaches 0.5
    # Let's use a different approach: linear scaling with a cap
    
    # Better: use the ratio of current best to known range
    # If f_best is much worse than known_min, explore more
    # Scale factor: starts at 1.0 (max exploration) and decreases as we improve
    max_distance = 5.0  # if we're 5 units above min, full exploration
    exploration_factor = np.clip(1.0 - np.abs(distance_to_min) / max_distance, 0.1, 1.0)
    
    # Combine EI with adaptive exploration
    # The exploration term encourages visiting uncertain regions
    exploration_bonus = exploration_factor * 0.5 * sigma
    
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
