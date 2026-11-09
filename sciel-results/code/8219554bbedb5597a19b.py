"""Cost-aware acquisition function for Bayesian optimization on 3-hump camel 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline tracking
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 10s safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviations
    f_best : float - best observed value so far
    cost : array of shape (2000,) - evaluation costs for each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = more desirable)
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        # Return a simple, fast fallback
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        cost_safe = np.maximum(cost, 1e-9)
        return ei / np.sqrt(cost_safe)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # --- Expected Improvement (for minimization) ---
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # --- Lower-bound regret term (encourages exploration) ---
    # The minimum possible value in a region is roughly mu - sigma.
    # Regret if we don't explore = f_best - (mu - sigma) = (f_best - mu) + sigma.
    # This is positive whenever there's uncertainty, even when EI == 0.
    potential = (f_best - mu) + sigma
    # Normalize potential using f_best as a scale reference
    scale = max(abs(f_best), 1e-6)
    potential_norm = np.clip(potential / (scale + 1e-9), 0.0, 10.0)
    
    # --- Cost model ---
    # cost = 1 + mean(x_scaled) in [1, 2]. Cheaper near the center of the box.
    # We want to encourage cheap points but not so much that we ignore the objective.
    cheapness = np.clip(1.0 - (cost - 1.0), 0.0, 1.0)
    
    # --- Cost-aware normalization ---
    # Use sqrt(cost) instead of cost to be gentler
    ei_cost_aware = ei / np.sqrt(cost)
    exploration_cost_aware = potential_norm / np.sqrt(cost)
    
    # --- Dynamic blend based on f_best ---
    # When f_best is high (early), EI values are large and well-separated, so EI-based selection works well.
    # When f_best is low (late), EI values are small and noisy, so we need exploration to find the global min.
    f_best_abs = abs(f_best)
    # weight_explore ~ 1 when f_best < 0 (impossible, but safe), ~ 0 when f_best > 10.
    weight_explore = np.clip(1.0 - f_best_abs / 10.0, 0.0, 1.0)
    
    # Combine EI and exploration with dynamic weights
    # - EI dominates when there's clear improvement.
    # - Exploration kicks in when EI is small (uncertainty remains).
    # - Cheapness is a minor tiebreaker.
    score = (1.0 - weight_explore) * ei_cost_aware + weight_explore * exploration_cost_aware
    
    # Add a small cheapness bonus to maximize budget usage
    cheap_bonus = 0.05 * cheapness
    score += cheap_bonus
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Ensure non-negative (higher = better)
    score = np.maximum(score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
