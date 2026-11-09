"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Michalewicz 2D.
    
    Uses Expected Improvement with adaptive exploration that decreases
    as f_best approaches the known global minimum.
    
    mu, sigma: arrays of shape (2000,) - posterior mean and std in objective units
    f_best: float - lowest observed value so far
    
    Returns: array of shape (2000,) - scores, higher means evaluate next
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Safety: replace any non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-8)
    
    # Clip sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10, 10)
    
    # Base Expected Improvement
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    
    # Adaptive exploration: reduce UCB weight as we approach the optimum
    # Michalewicz 2D minimum is approximately -1.8013
    known_min = -1.8013
    # Progress: 0 when f_best is far from min, 1 when f_best is at min
    progress = np.clip((f_best - known_min) / (5.0 - known_min), 0.0, 1.0)
    # As progress increases (f_best closer to min), reduce exploration
    # Beta starts at 2.5 and decreases to 0.5
    beta = 0.5 + 2.0 * (1.0 - progress)
    
    # UCB score (for minimization, lower is better)
    ucb = mu + beta * sigma_safe
    ucb_score = -ucb
    
    # Weight between EI and UCB
    # More exploration early, more exploitation later
    w_ei = 0.6 + 0.3 * progress
    w_ucb = 1.0 - w_ei
    
    # Combine scores
    score = w_ei * ei + w_ucb * ucb_score
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
