"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using a calibrated exploitation term plus relative uncertainty.
    
    Score = (-(mu) / cost) + alpha * (sigma / cost) * log(1 + 1/sigma_scale)
    
    This lets the search commit to promising low-mean regions while still exploring,
    which pure EI/cost under-exploits in a cheap-to-evaluate 2-D landscape.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions for 2000 candidates
    sigma : array-like, shape (2000,)
        GP standard deviation predictions for 2000 candidates
    f_best : float
        Best observed value so far
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive and handle numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Deadline guard: track elapsed time
    t_start = time.time()
    
    # Calibration: scale sigma relative to typical scale
    # Use median sigma as a reference scale
    sigma_scale = np.median(sigma_safe)
    if sigma_scale < 1e-9:
        sigma_scale = 1e-9
    
    # Exploitation term: -mu / cost (lower mu is better, divide by cost)
    exploitation = -mu / cost_safe
    
    # Exploration term: sigma / cost, scaled logarithmically to avoid over-exploration
    # Use log(1 + 1/sigma_rel) where sigma_rel = sigma / sigma_scale
    sigma_rel = sigma_safe / sigma_scale
    # For large sigma_rel, log(1 + 1/sigma_rel) -> 0, so exploration is damped
    # For small sigma_rel, log(1 + 1/sigma_rel) -> log(2), moderate exploration
    exploration_weight = np.log1p(1.0 / np.maximum(sigma_rel, 1e-9))
    exploration = (sigma_safe / cost_safe) * exploration_weight
    
    # Combine: exploitation dominates, exploration is a secondary signal
    # alpha controls exploration strength
    alpha = 1.5
    scores = exploitation + alpha * exploration
    
    # Check deadline
    t_elapsed = time.time() - t_start
    if t_elapsed > 280:
        # Return best result found with safety margin
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
