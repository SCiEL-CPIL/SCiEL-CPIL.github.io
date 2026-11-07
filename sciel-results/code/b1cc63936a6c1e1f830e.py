"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = None


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Ackley 2D.
    
    Uses a combination of Expected Improvement (EI) and an adaptive Upper Confidence Bound (UCB)
    to balance exploitation and exploration. The exploration weight (beta) and the
    EI/UCB mix (alpha) are adapted based on the progress of the optimization (f_best).
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores. Higher values indicate more promising candidates.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: return immediately if we're close to the 330s limit
    deadline = 330.0 - 2.0  # safety margin
    if time.time() > deadline:
        # Fallback: return a simple score if we're out of time (should not happen for vectorized ops)
        return -mu + sigma
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI) for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive exploration: when f_best is still high, explore more;
    # when f_best is close to 0, exploit more.
    # f_best ranges from ~10 (Ackley max) down to 0 (minimum).
    # Map f_best to a progress indicator in [0, 1].
    progress = np.clip(1.0 - f_best / 10.0, 0.0, 1.0)
    
    # Interpolate beta: start at 1.5 (more exploration) -> end at 0.7 (more exploitation)
    beta = 1.5 - 0.8 * progress
    
    # Interpolate alpha: start at 0.75 (more UCB) -> end at 0.95 (more EI)
    alpha = 0.75 + 0.20 * progress
    
    # Upper Confidence Bound (UCB) for minimization
    ucb = -mu + beta * sigma_safe
    
    # Combine EI and UCB with a weighted average
    scores = alpha * ei + (1.0 - alpha) * ucb
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
