"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_TIME_LIMIT = 320.0  # safety margin before 330s kill


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points.
    
    Uses a hybrid approach combining Expected Improvement (EI) with
    a penalty for points where sigma is very small (exploited regions)
    and a bonus for points with moderate uncertainty (exploration).
    
    Parameters:
        mu: array of shape (2000,) - GP mean at candidates
        sigma: array of shape (2000,) - GP std dev at candidates
        f_best: float - best observed value so far
    
    Returns:
        array of shape (2000,) - acquisition scores (higher = better)
    """
    # Deadline guard
    if time.time() - _start_time > _TIME_LIMIT:
        # If we're close to timeout, just return EI quickly
        z = (f_best - mu) / (sigma + 1e-12)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Additional exploration term: encourage exploring regions with
    # moderate uncertainty that haven't been fully exploited
    # This helps escape local optima in the 1D sphere
    
    # Scale uncertainty relative to the range of sigma
    sigma_min = np.min(sigma_safe)
    sigma_max = np.max(sigma_safe)
    sigma_range = sigma_max - sigma_min + 1e-12
    
    # Normalized uncertainty (0 to 1)
    sigma_norm = (sigma_safe - sigma_min) / sigma_range
    
    # Exploration bonus: peaks at moderate uncertainty
    # Use a Gaussian-like bump centered at 0.5 normalized uncertainty
    exploration_bonus = 0.1 * np.exp(-((sigma_norm - 0.3) ** 2) / 0.1)
    
    # Combine EI with exploration
    # Scale EI to have comparable magnitude to exploration bonus
    ei_max = np.max(ei) + 1e-12
    ei_normalized = ei / ei_max
    
    # Final score: weighted combination
    # Higher weight on EI when f_best is small (close to optimum)
    # More exploration when f_best is large (early in optimization)
    weight_exploration = np.clip(f_best / (f_best + 1.0), 0.1, 0.5)
    
    scores = (1.0 - weight_exploration) * ei_normalized + weight_exploration * exploration_bonus
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
