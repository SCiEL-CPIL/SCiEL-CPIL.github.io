"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of hartmann 3d.
    
    Uses a combination of Expected Improvement (EI) and a UCB-like term
    to balance exploitation and exploration. The UCB term is scaled by
    the standard deviation to encourage exploration in uncertain regions.
    
    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Mean of the GP posterior at candidate points.
    sigma : np.ndarray, shape (2000,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Deadline guard: track elapsed time
    t_start = time.time()
    deadline = t_start + 290.0  # 10s safety margin before 300s limit
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score for Expected Improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (for minimization)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # UCB-like exploration term: encourages exploration where uncertainty is high
    # Use a time-varying or fixed beta to balance exploration/exploitation
    # For 30 evaluations total (5 initial + 25 BO steps), use moderate exploration
    beta = 2.0  # Exploration parameter
    
    # UCB term: -mu + beta * sigma (since we're minimizing, lower mu is better)
    ucb = -mu + beta * sigma_safe
    
    # Combine EI and UCB: weight EI more heavily as it's a proven acquisition
    # Use a blended score that emphasizes EI but includes exploration
    scores = 0.7 * ei + 0.3 * ucb
    
    # Check deadline
    if time.time() > deadline:
        # Return the best result found so far (shouldn't happen in normal operation)
        return scores
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
