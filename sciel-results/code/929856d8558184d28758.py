"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_module_start = time.time()
_DEADLINE = 280.0  # 20s safety margin before 300s limit


def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() - _module_start > _DEADLINE:
        return -mu.astype(float)
    
    # Structural change: Use a hybrid of Expected Improvement and a UCB-like term
    # to balance exploitation and exploration more aggressively than pure EI or POI.
    # This creates a different search dynamic that may escape local minima better.
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.where(sigma > 1e-10, sigma, 1e-10)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate Phi(z) and phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Calculate Expected Improvement (EI)
    # EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Calculate Upper Confidence Bound (UCB) component
    # beta controls exploration-exploitation trade-off
    # A dynamic beta that increases slightly with evaluation count could be ideal,
    # but we don't have access to step count. We use a fixed beta tuned for this problem.
    beta = 2.0  # Exploration weight
    ucb = -mu + beta * sigma_safe
    
    # Normalize EI and UCB to [0, 1] range for combination
    # This prevents scale issues if f_best is large
    ei_max = np.max(ei) if np.max(ei) > 1e-10 else 1.0
    ei_norm = ei / ei_max
    
    ucb_max = np.max(ucb) if np.max(ucb) > 1e-10 else 1.0
    ucb_min = np.min(ucb) if np.min(ucb) < -1e-10 else -1.0
    ucb_range = ucb_max - ucb_min if ucb_range > 1e-10 else 1.0
    ucb_norm = (ucb - ucb_min) / ucb_range
    
    # Combine: Weighted sum of normalized EI and UCB
    # 0.6 * EI + 0.4 * UCB gives a balance favoring improvement but with strong exploration
    scores = 0.6 * ei_norm + 0.4 * ucb_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
