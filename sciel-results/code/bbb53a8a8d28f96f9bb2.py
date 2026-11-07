"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: track elapsed time to avoid timeout
_t_start = time.time()
_DEADLINE = 325.0  # 5 second safety margin before 330s kill

def _elapsed():
    return time.time() - _t_start

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP mean predictions for 2000 candidate points.
    sigma : array of shape (2000,)
        GP standard deviation predictions for 2000 candidate points.
    f_best : float
        Best observed value so far.
    cost : array of shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores (higher = better candidate to evaluate).
    """
    # Deadline guard
    if _elapsed() > _DEADLINE:
        return np.zeros_like(mu)
    
    n = len(mu)
    
    # Handle edge cases: ensure sigma is positive and finite
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 1e-9), sigma, 1e-9)
    mu_safe = np.where(np.isfinite(mu), mu, f_best)
    
    # Compute Expected Improvement (EI)
    z = (f_best - mu_safe) / sigma_safe
    ei = (f_best - mu_safe) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute Probability of Improvement (PI)
    pi = norm.cdf(z)
    
    # Compute a hybrid acquisition that balances EI and PI
    # This helps in early stages when EI might be dominated by noise
    # and in later stages when we want to exploit
    
    # Compute a normalized uncertainty bonus
    sigma_norm = sigma_safe / (np.max(sigma_safe) + 1e-9)
    
    # Weighted combination: EI dominates, but PI and uncertainty help
    # Use a soft cost-aware penalty
    cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Cost-aware scaling: divide by cost^alpha where alpha < 1 to be less aggressive
    cost_factor = cost_safe ** 0.5
    
    # Base score: EI / cost_factor
    base_score = ei / cost_factor
    
    # Add a small PI component to encourage exploration in uncertain regions
    # and a small uncertainty bonus
    exploration_bonus = 0.1 * pi * sigma_norm
    
    # Final score
    scores = base_score + exploration_bonus
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure non-negative (EI should be non-negative, but numerical issues might cause small negatives)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
