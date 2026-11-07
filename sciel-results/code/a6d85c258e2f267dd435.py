"""Acquisition function for Bayesian optimization of sphere 1d."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for minimizing sphere 1d on [-5, 5].
    Uses a hybrid approach: Early on, use aggressive UCB exploration.
    Later, use Expected Improvement for exploitation.
    """
    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # The sphere function f(x) = x^2 on [-5, 5] has a maximum value of 25.
    f_max_possible = 25.0
    
    # Calculate progress: 0 at start (f_best ~ 25), 1 at end (f_best ~ 0)
    f_best_clipped = np.clip(f_best, 0.0, f_max_possible)
    progress = 1.0 - (f_best_clipped / f_max_possible)
    
    # Adaptive Beta (exploration parameter)
    # Start with high exploration and reduce to low exploration
    beta_high = 2.5
    beta_low = 0.2
    beta = beta_high - (beta_high - beta_low) * progress
    
    # Compute UCB score for minimization
    ucb_score = -mu + beta * sigma_safe
    
    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    from scipy.stats import norm
    z = (f_best - mu) / sigma_safe
    ei_score = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Blend UCB and EI based on progress
    # Early on (low progress), rely more on UCB for exploration
    # Later (high progress), rely more on EI for exploitation
    # Use a smooth transition: weight for UCB decreases, weight for EI increases
    ucb_weight = 1.0 - 0.8 * progress  # From 1.0 to 0.2
    ei_weight = 0.8 * progress         # From 0.0 to 0.8
    
    # Normalize scores to be on comparable scales
    # Center and scale each score set
    ucb_centered = ucb_score - np.mean(ucb_score)
    ucb_std = np.std(ucb_score) + 1e-8
    ucb_normalized = ucb_centered / ucb_std
    
    ei_centered = ei_score - np.mean(ei_score)
    ei_std = np.std(ei_score) + 1e-8
    ei_normalized = ei_centered / ei_std
    
    # Combine the normalized scores
    score = ucb_weight * ucb_normalized + ei_weight * ei_normalized
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
