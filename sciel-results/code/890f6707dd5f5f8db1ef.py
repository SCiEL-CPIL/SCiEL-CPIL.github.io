"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global start time for deadline guard
_START_TIME = time.time()
_TIMEOUT = 290  # Safety margin before 300s


def acquisition(mu, sigma, f_best):
    """
    Acquisition function that blends Expected Improvement (EI) with Upper Confidence Bound (UCB).
    The blend adapts based on local uncertainty, favoring exploration (UCB) in high-uncertainty
    regions and exploitation (EI) in low-uncertainty regions.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        Posterior mean of the GP at candidate points.
    sigma : array-like, shape (2000,)
        Posterior standard deviation of the GP at candidate points.
    f_best : float
        Best observed function value so far (minimum).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Check deadline - if we're running out of time, return a fast fallback
    if time.time() - _START_TIME > _TIMEOUT:
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        return ei

    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)

    # Compute Expected Improvement for minimization
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Compute UCB for minimization: score = -mu + beta * sigma
    # We use a fixed moderate beta, as the adaptive weighting below handles the tradeoff
    beta = 2.0
    ucb = -mu + beta * sigma_safe

    # Normalize EI and UCB to [0, 1] range for comparable blending
    ei_min = np.min(ei)
    ei_max = np.max(ei)
    ei_range = max(ei_max - ei_min, 1e-9)
    ei_norm = (ei - ei_min) / ei_range

    ucb_min = np.min(ucb)
    ucb_max = np.max(ucb)
    ucb_range = max(ucb_max - ucb_min, 1e-9)
    ucb_norm = (ucb - ucb_min) / ucb_range

    # Adaptive Blending:
    # High uncertainty (high sigma) -> Favor UCB (Exploration)
    # Low uncertainty (low sigma) -> Favor EI (Exploitation)
    
    # Calculate relative uncertainty for each point
    sigma_mean = np.mean(sigma_safe)
    if sigma_mean < 1e-9:
        sigma_mean = 1e-9
        
    # exploration_weight is high where sigma is high
    # Clip to avoid extreme outliers dominating
    exploration_weight = np.clip(sigma_safe / (2.0 * sigma_mean), 0.0, 1.0)
    
    # Blend coefficients
    # When exploration_weight is 1 (high sigma), we want high UCB weight.
    # When exploration_weight is 0 (low sigma), we want high EI weight.
    
    # Let's say EI weight goes from 0.8 (low sigma) to 0.2 (high sigma)
    ei_weight = 0.8 - 0.6 * exploration_weight
    ucb_weight = 1.0 - ei_weight
    
    scores = ei_weight * ei_norm + ucb_weight * ucb_norm

    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
