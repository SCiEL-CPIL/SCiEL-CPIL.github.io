"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of branin 2d.
    
    Uses a Kriging Believer style approach:
    1. Compute UCB scores to identify promising regions (high uncertainty + low mean)
    2. Among candidates in the top UCB regions, score by expected improvement potential
       (negative mean, favoring low predicted values)
    This decouples exploration (via UCB) from exploitation (via low predicted values).
    
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
    
    # Step 1: Compute UCB scores for exploration
    # UCB = mu - beta * sigma (for minimization, we want low mu and high sigma)
    # Normalize sigma to [0, 1] range for stable beta scaling
    sigma_min = np.min(sigma_safe)
    sigma_range = np.max(sigma_safe) - sigma_min
    if sigma_range > 1e-12:
        sigma_norm = (sigma_safe - sigma_min) / sigma_range
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Dynamic beta: start with moderate exploration, reduce as we find better values
    # Use a fixed beta for stability
    beta = 1.5
    
    ucb_scores = mu - beta * sigma_safe
    
    # Step 2: Identify top UCB candidates (exploration regions)
    # Take top 20% of candidates by UCB score
    n_top = max(1, int(0.2 * len(ucb_scores)))
    top_indices = np.argpartition(-ucb_scores, n_top)[:n_top]
    
    # Step 3: Among top UCB candidates, score by exploitation potential
    # Use Expected Improvement (EI) for a principled exploitation score
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei_scores = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Step 4: Combine - only give high scores to candidates in top UCB regions
    # For candidates outside top UCB, use a lower weight on EI
    scores = np.zeros_like(ei_scores)
    scores[top_indices] = ei_scores[top_indices]
    
    # For non-top candidates, use a reduced EI score to allow some exploration
    # outside the top UCB regions
    non_top_mask = ~np.isin(np.arange(len(scores)), top_indices)
    scores[non_top_mask] = 0.3 * ei_scores[non_top_mask]
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - SAFETY_MARGIN:
        # Return current scores; they should already be computed
        pass
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
