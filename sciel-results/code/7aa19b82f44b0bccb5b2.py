"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


# Deadline guard: keep explicit wall-clock tracking
_START_TIME = time.time()
_TIME_LIMIT = 300.0  # evaluator kills at 330s, leave margin
_TIME_MARGIN = 10.0  # safety margin seconds


def _time_left():
    return _TIME_LIMIT - _TIME_MARGIN - (time.time() - _START_TIME)


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Branin 2D.
    
    Uses an adaptive combination of Expected Improvement (EI) and 
    multi-scale UCB to balance exploitation and exploration for
    the multi-modal Branin function.
    
    Parameters:
    -----------
    mu : ndarray, shape (n,)
        GP mean at candidate points (objective units)
    sigma : ndarray, shape (n,)
        GP standard deviation at candidate points (objective units)
    f_best : float
        Best observed value so far
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores (higher = more promising)
    """
    # Check time budget - if we're running out, return a simple fallback
    if _time_left() < 1.0:
        # Fallback: pure EI
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Compute EI normalization
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    # Multi-scale UCB: try several beta values and take a weighted combination
    # This acts as a soft search over exploration levels
    betas = np.array([0.5, 1.0, 2.0, 4.0])
    
    # Compute UCB scores for each beta: -mu + beta * sigma (higher is better for minimization)
    ucb_scores = -mu[None, :] + betas[:, None] * sigma_safe[None, :]
    
    # Normalize each beta's UCB scores to [0, 1]
    ucb_scores_norm = np.zeros_like(ucb_scores)
    for i in range(len(betas)):
        row = ucb_scores[i, :]
        row_max = np.max(row)
        row_min = np.min(row)
        if row_max - row_min > 1e-12:
            ucb_scores_norm[i, :] = (row - row_min) / (row_max - row_min)
    
    # Weight the UCB scores: favor moderate exploration
    # Early on, we want more exploration; later, more exploitation
    # Use the ratio of sigma to mu spread as a heuristic for "how much we've learned"
    mu_std = np.std(mu)
    sigma_mean = np.mean(sigma_safe)
    
    # If sigma is large relative to mu spread, we're early -> more exploration
    # If sigma is small relative to mu spread, we're later -> more exploitation
    if mu_std > 1e-12:
        exploration_ratio = sigma_mean / mu_std
    else:
        exploration_ratio = 1.0
    
    # Clamp the ratio to [0.1, 2.0]
    exploration_ratio = np.clip(exploration_ratio, 0.1, 2.0)
    
    # Weight the betas: higher exploration_ratio -> weight higher betas more
    # beta weights: [0.5, 1.0, 2.0, 4.0]
    if exploration_ratio > 1.0:
        # More exploration needed
        beta_weights = np.array([0.1, 0.2, 0.3, 0.4])
    else:
        # Less exploration needed
        beta_weights = np.array([0.4, 0.3, 0.2, 0.1])
    
    beta_weights = beta_weights / np.sum(beta_weights)
    
    # Weighted combination of UCB scores
    ucb_combined = np.sum(ucb_scores_norm * beta_weights[:, None], axis=0)
    
    # Final score: blend EI and UCB
    # Use 0.75 EI + 0.25 UCB for a balance that works well for Branin
    scale = max(ei_max, 1e-9)
    score = 0.75 * ei + 0.25 * ucb_combined * scale
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
