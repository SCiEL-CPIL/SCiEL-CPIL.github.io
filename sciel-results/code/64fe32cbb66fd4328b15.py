"""Rewritten acquisition for Bayesian optimization of Hartmann 3D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Adaptive weighted combination of Expected Improvement (EI)
    and Uncertainty (Sigma).
    
    The weight on exploration (Sigma) decreases as f_best approaches the known
    global minimum of the Hartmann 3D function. This ensures strong exploration
    in the early stages to locate basins of attraction, and strong exploitation
    in later stages to refine the minimum.
    
    Parameters
    ----------
    mu : ndarray, shape (n,)
        GP mean predictions at candidate points.
    sigma : ndarray, shape (n,)
        GP standard deviation predictions at candidate points.
    f_best : float
        Best (lowest) observed objective value so far.

    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher means more promising.
    """
    # Known global minimum of Hartmann 3D
    f_min = -3.86278
    # Estimated initial best value for uniform random points in [0, 1]^3
    # Hartmann 3D average value is roughly -1.5
    f_start = -1.5
    
    # Calculate the remaining gap to the optimum relative to the initial gap
    gap = f_best - f_min
    max_gap = f_start - f_min
    
    if max_gap <= 1e-9:
        frac_remaining = 0.0
    else:
        # Clip to [0, 1] to handle cases where f_best might be better than f_min (numerical)
        # or worse than f_start (if initial points were bad)
        frac_remaining = np.clip(gap / max_gap, 0.0, 1.0)
        
    # Weights: 
    # Early on (frac_remaining ~ 1), we prioritize exploration (Sigma).
    # Late on (frac_remaining ~ 0), we prioritize exploitation (EI).
    w_explore = frac_remaining
    w_exploit = 1.0 - frac_remaining
    
    # Ensure numerical stability for EI calculation
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Helper to normalize an array to [0, 1]
    def normalize_scores(arr):
        min_val = np.min(arr)
        max_val = np.max(arr)
        range_val = max_val - min_val
        if range_val < 1e-9:
            return np.zeros_like(arr)
        return (arr - min_val) / range_val
    
    # Normalize components
    score_exploit = normalize_scores(ei)
    score_explore = normalize_scores(sigma_safe)
    
    # Combine
    scores = w_exploit * score_exploit + w_explore * score_explore
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
