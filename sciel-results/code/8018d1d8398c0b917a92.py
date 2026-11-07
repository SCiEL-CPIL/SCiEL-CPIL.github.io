"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    
    Uses Probability of Improvement (PI) as the primary signal.
    PI is effective for Branin 2D because it balances exploration and
    exploitation well without over-weighting deep but certain valleys,
    helping to explore multiple basins of attraction in the multimodal landscape.
    A small normalized sigma bonus is added to differentiate candidates
    with similar PI, encouraging mild exploration in uncertain regions.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Mean of the GP posterior at candidate points (in objective units).
    sigma : array-like, shape (2000,)
        Standard deviation of the GP posterior at candidate points (in objective units).
    f_best : float
        Best observed function value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores for each candidate point. Higher means evaluate next.
    """
    start_time = time.time()
    
    # Safety margin: ensure we finish well before the 300s limit
    DEADLINE = 300.0
    SAFETY_MARGIN = 10.0
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite to avoid division by zero or negative variance
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute z-score for Probability of Improvement
    # z = (f_best - mu) / sigma
    # P(f(x) < f_best) = Phi(z)
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement
    pi_scores = norm.cdf(z)
    
    # Exploration bonus: small weight on normalized sigma
    # This helps select among candidates with similar PI by preferring
    # slightly more uncertain ones, which can help escape local minima.
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-12:
        exploration_bonus = 0.05 * (sigma_safe / sigma_max)
    else:
        exploration_bonus = np.zeros_like(sigma_safe)
    
    # Combine: primarily PI with small exploration bonus
    scores = pi_scores + exploration_bonus
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - SAFETY_MARGIN:
        # Return current scores; they should already be computed
        pass
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
