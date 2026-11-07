"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of beale 2d.
    
    Uses a hybrid approach: primarily Expected Improvement (EI) for exploitation
    with a small UCB component for exploration, carefully balanced to avoid
    over-exploration in later stages.
    
    Parameters:
    -----------
    mu : array-like, shape (n,)
        Mean of the Gaussian process posterior at candidate points.
    sigma : array-like, shape (n,)
        Standard deviation of the Gaussian process posterior at candidate points.
    f_best : float
        Best observed value so far (lowest, since we're minimizing).
    
    Returns:
    --------
    scores : array-like, shape (n,)
        Acquisition scores; higher means more promising to evaluate next.
    """
    start_time = time.time()
    deadline = 320.0  # Safety margin before 330s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)  # Clip for numerical stability
    
    # Expected Improvement (EI) - higher is better
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Check deadline early
    if time.time() > deadline:
        # Return EI alone as fallback if we're running low on time
        return np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    # UCB term for exploration - higher is better
    # Use a moderate beta to balance exploration/exploitation
    # The key insight: use a smaller beta than the current program to favor exploitation
    beta = 0.3
    ucb = -mu + beta * sigma
    
    # Combine EI and UCB with appropriate weights
    # Normalize both to have similar scales
    ei_max = np.max(ei)
    ei_min = np.min(ei)
    ei_range = ei_max - ei_min + 1e-10
    ei_norm = (ei - ei_min) / ei_range
    
    ucb_max = np.max(ucb)
    ucb_min = np.min(ucb)
    ucb_range = ucb_max - ucb_min + 1e-10
    ucb_norm = (ucb - ucb_min) / ucb_range
    
    # Weighted combination: primarily EI, with less UCB for exploration
    # The insight: reduce the UCB weight to favor exploitation more strongly
    scores = 0.85 * ei_norm + 0.15 * ucb_norm
    
    # Check deadline
    if time.time() > deadline:
        # Return EI alone as fallback if we're running low on time
        return np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
# EVOLVE-BLOCK-END
