"""Rewritten acquisition for cost-aware Bayesian optimization on Hartmann 3D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition blending normalized Expected Improvement (EI)
    and Probability of Improvement (PI) to balance exploitation and exploration.
    
    Parameters:
    -----------
    mu : ndarray, shape (N,)
        GP mean predictions in objective units.
    sigma : ndarray, shape (N,)
        GP standard deviation predictions in objective units.
    f_best : float
        Best (lowest) observed objective value.
    cost : ndarray, shape (N,)
        Cost of evaluating each candidate (higher is worse).
    
    Returns:
    --------
    scores : ndarray, shape (N,)
        Acquisition scores; higher means more desirable to evaluate.
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)) or not np.all(np.isfinite(cost)):
        return np.zeros_like(mu)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-score for EI/PI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Blend EI and PI: use a mix that favors exploration in uncertain regions
    # Normalize EI by sigma to make it scale-invariant, then blend with PI
    # The blend weight can be tuned; here we use 0.5*normalized_EI + 0.5*PI
    # But a more principled approach: use EI directly but add a small exploration bonus
    # based on uncertainty (sigma) relative to the spread of mu
    
    # Alternative: use a weighted combination where the weight on exploration
    # increases when sigma is large relative to the range of mu
    mu_min = np.min(mu)
    mu_max = np.max(mu)
    mu_range = mu_max - mu_min + 1e-9
    
    # Exploration bonus: proportional to sigma / mu_range, capped
    exploration_bonus = 0.1 * (sigma_safe / mu_range)
    
    # Combined acquisition: cost-normalized EI + exploration bonus
    # The exploration bonus encourages evaluating uncertain points
    score = ei / cost_safe + exploration_bonus / cost_safe
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
