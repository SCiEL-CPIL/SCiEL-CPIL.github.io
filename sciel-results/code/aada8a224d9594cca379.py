"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for minimizing shekel 4d.
    
    Uses a robust cost-aware EI with adaptive exploration.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        GP mean predictions (in objective units).
    sigma : ndarray, shape (2000,)
        GP standard deviation predictions (in objective units).
    f_best : float
        Lowest observed function value.
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    t0 = time.time()
    DEADLINE = 310.0  # seconds, safety margin before 330s kill
    
    # Handle edge cases
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-9)
    
    # --- Compute Expected Improvement (EI) ---
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    delta = f_best - mu
    z = delta / sigma_safe
    z_clipped = np.clip(z, -50, 50)
    
    ei = delta * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # --- Compute UCB for exploration ---
    # Use a beta that adapts to the uncertainty landscape
    # Higher beta when overall uncertainty is high
    sigma_median = np.median(sigma_safe)
    sigma_max = np.max(sigma_safe)
    
    # Beta should be moderate: not too aggressive exploration
    # Scale with typical uncertainty relative to the range
    if sigma_max > 1e-8:
        uncertainty_ratio = sigma_median / sigma_max
    else:
        uncertainty_ratio = 0.0
    
    # Beta in [0.5, 2.0] range
    beta = 0.5 + 1.5 * uncertainty_ratio
    
    ucb = mu + beta * sigma_safe
    
    # --- Combine EI and UCB ---
    # Normalize each component to [0, 1] range for blending
    ei_max = np.max(ei) if np.max(ei) > 1e-10 else 1.0
    ei_norm = ei / ei_max
    
    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    if ucb_max - ucb_min > 1e-10:
        ucb_norm = (ucb - ucb_min) / (ucb_max - ucb_min)
    else:
        ucb_norm = np.zeros_like(ucb)
    
    # Blend: weight exploitation more as we get more confident
    # Use average uncertainty as a global confidence measure
    avg_sigma = np.mean(sigma_safe)
    if sigma_max > 1e-8:
        avg_uncertainty = avg_sigma / sigma_max
    else:
        avg_uncertainty = 0.0
    
    # When uncertainty is high, weight exploration more
    w_explore = 0.2 + 0.5 * avg_uncertainty
    w_exploit = 1.0 - w_explore
    
    combined = w_exploit * ei_norm + w_explore * ucb_norm
    
    # --- Cost-aware penalty ---
    # Penalize expensive points, but less aggressively in high-uncertainty regions
    # Use a soft penalty: score / (1 + cost_penalty)
    # cost_penalty scales with cost but is damped by uncertainty
    
    # Local uncertainty: high sigma means we can afford higher cost
    sigma_norm = sigma_safe / (sigma_max + 1e-8)
    
    # Cost penalty: lower penalty when uncertainty is high
    # penalty_factor = cost * (1 - 0.5 * sigma_norm)
    # This reduces the effective cost in high-uncertainty regions
    penalty_factor = cost_safe * (1.0 - 0.4 * sigma_norm)
    penalty_factor = np.maximum(penalty_factor, 0.1)  # Ensure minimum penalty
    
    scores = combined / penalty_factor
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline check
    if time.time() - t0 > DEADLINE:
        return scores
    
    return scores
# EVOLVE-BLOCK-END
