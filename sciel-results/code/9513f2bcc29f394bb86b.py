"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of hartmann 6d.
    Uses a hybrid of Expected Improvement (EI) and Upper Confidence Bound (UCB)
    to balance exploitation and exploration.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        Posterior mean at candidate points.
    sigma : array_like, shape (2000,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best observed value so far.
    
    Returns:
    --------
    scores : array_like, shape (2000,)
        Acquisition scores, higher is better.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle edge cases where sigma might be zero or very small
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute UCB score: -mu + beta * sigma (for minimization, we want low mu and high sigma)
    # Since we're minimizing, high UCB score means promising region
    # Standard UCB for minimization: -mu + beta*sigma
    # But we want higher score to be better, so we can use: -mu + beta*sigma
    # Actually, for minimization, we want to find low values.
    # EI naturally handles this. Let's combine EI with a bonus for high uncertainty.
    
    # Hybrid approach: EI + exploration bonus
    # The exploration bonus encourages visiting regions with high sigma
    # This helps escape local minima in multimodal landscapes like Hartmann 6D
    
    # Normalize sigma to [0, 1] for the bonus term
    sigma_max = np.max(sigma_safe) if np.max(sigma_safe) > 0 else 1.0
    sigma_min = np.min(sigma_safe)
    sigma_range = sigma_max - sigma_min
    if sigma_range > 0:
        sigma_norm = (sigma_safe - sigma_min) / sigma_range
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Weight the exploration bonus
    # Use a moderate exploration weight that balances exploitation and exploration
    exploration_weight = 0.5
    
    # Combined score: EI + exploration_weight * sigma_norm * (f_best - min(mu))
    # This ensures that high uncertainty regions get a boost proportional to potential improvement
    mu_min = np.min(mu)
    potential_improvement = np.maximum(f_best - mu, 0)
    
    # Scale the exploration bonus by potential improvement
    exploration_bonus = exploration_weight * sigma_norm * potential_improvement
    
    scores = ei + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
