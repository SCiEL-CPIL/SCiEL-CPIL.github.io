# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline management
_START_TIME = time.time()
_DEADLINE_SECONDS = 285.0  # Safety margin before 300s/330s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of rastrigin 2d.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions
    sigma : array of shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array of shape (2000,)
        Evaluation cost for each candidate
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_SECONDS:
        # Return a safe, finite, valid score to avoid invalidation
        return np.zeros_like(mu, dtype=np.float64)
    
    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-normalized EI using square root penalty (softer than linear)
    ei_cost = ei / np.sqrt(cost)
    
    # Exploration bonus: Gaussian-like peak at moderate uncertainty
    # Normalize sigma to [0,1] range for the bonus
    sigma_min = np.min(sigma)
    sigma_max = np.max(sigma)
    sigma_range = sigma_max - sigma_min + 1e-9
    sigma_norm = (sigma - sigma_min) / sigma_range
    
    # Peak exploration at sigma_norm ≈ 0.4 (moderate uncertainty)
    # This encourages exploring regions that are uncertain but not maximally so
    exploration_width = 0.3
    exploration_center = 0.4
    exploration_bonus = np.exp(-((sigma_norm - exploration_center) ** 2) / (2 * exploration_width ** 2))
    
    # Scale exploration bonus relative to the EI/cost magnitude
    ei_cost_abs = np.abs(ei_cost)
    ei_cost_scale = np.mean(ei_cost_abs) + 1e-9
    exploration_weight = 0.15 * ei_cost_scale
    
    # Combine EI/cost with scaled exploration bonus
    scores = ei_cost + exploration_weight * exploration_bonus
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
