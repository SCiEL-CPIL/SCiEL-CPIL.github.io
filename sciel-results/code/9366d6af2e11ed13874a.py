# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline management
_START_TIME = time.monotonic()
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
    elapsed = time.monotonic() - _START_TIME
    if elapsed > _DEADLINE_SECONDS:
        # Return a safe, finite, valid score to avoid invalidation
        return np.zeros_like(mu, dtype=np.float64)
    
    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle NaN/Inf in inputs
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=-1e6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e-9)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # Expected Improvement (EI)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-normalized EI using square root penalty (softer than linear)
    # This allows more exploration in moderately expensive regions
    ei_cost = ei / np.sqrt(cost)
    
    # UCB component for exploration
    beta = 2.0
    ucb = mu + beta * sigma
    ucb_cost = (f_best - ucb) / np.sqrt(cost)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    pi_cost = pi / np.sqrt(cost)
    
    # Blend EI and UCB with adaptive weights
    # When uncertainty is high, weight UCB more; when low, weight EI more
    sigma_max = np.max(sigma)
    sigma_min = np.min(sigma)
    sigma_range = sigma_max - sigma_min + 1e-9
    sigma_norm = (sigma - sigma_min) / sigma_range
    
    # Adaptive blend: more UCB when sigma is high
    alpha = 0.3 + 0.4 * sigma_norm  # ranges from 0.3 to 0.7
    
    # Combined acquisition
    base_score = alpha * ucb_cost + (1 - alpha) * ei_cost
    
    # Add a small PI component to encourage exploitation of likely improvements
    base_score += 0.1 * pi_cost
    
    # Exploration bonus: Gaussian-like peak at moderate uncertainty
    # This encourages exploring regions with moderate uncertainty
    sigma_center = 0.4
    sigma_width = 0.3
    exploration_bonus = 0.15 * np.exp(-0.5 * ((sigma_norm - sigma_center) / sigma_width) ** 2)
    
    # Final score
    scores = base_score + exploration_bonus
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
