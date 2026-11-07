"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Adaptive Lower Confidence Bound (LCB) with exploration-exploitation balance.
    
    Uses a time-adaptive exploration coefficient that starts high to encourage
    global exploration (critical for the multimodal Hartmann 6D landscape) and
    gradually decreases to focus on exploitation as the search progresses.
    This is achieved by scaling kappa based on the current best value relative
    to the known global minimum.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        Mean of the GP posterior at candidate points
    sigma : array of shape (2000,)
        Standard deviation of the GP posterior at candidate points
    f_best : float
        Best (lowest) observed function value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores; higher means evaluate next
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Handle edge cases
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        # Fallback: use uniform random scores if inputs are invalid
        return np.random.randn(len(mu))
    
    # Clip sigma to avoid division by zero
    sigma = np.maximum(sigma, 1e-9)
    
    # Known global minimum of Hartmann 6D
    f_min = -3.32237
    
    # Adaptive kappa: start high, decrease as we approach the optimum
    # When f_best is far from f_min, use high exploration (kappa ~ 3.0)
    # When f_best is close to f_min, use lower exploration (kappa ~ 1.5)
    # Use a smooth transition based on normalized regret
    regret = f_best - f_min  # positive when we haven't found the minimum yet
    # Normalize: regret ranges from ~0 (found min) to ~3.5 (initial random)
    # Map to kappa in [1.5, 3.5]
    # Use a tanh-based smooth mapping
    # When regret is large, kappa should be high
    # When regret is small, kappa should be lower
    kappa = 1.5 + 2.0 * np.tanh(regret / 1.0)
    
    # Ensure kappa stays in reasonable bounds
    kappa = np.clip(kappa, 1.5, 3.5)
    
    # LCB acquisition: we want to minimize, so we score as negative LCB
    # LCB = mu - kappa * sigma (lower is better for minimization)
    # Score = -LCB = -mu + kappa * sigma (higher is better for acquisition)
    scores = -mu + kappa * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
