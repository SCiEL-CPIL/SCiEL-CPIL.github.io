"""Acquisition function for Bayesian optimization of styblinski tang 1d.
Uses Normalized GP-UCB with a dynamic exploration coefficient.
The exploration term is scaled by the median posterior standard deviation,
making the exploration pressure relative to the current state of knowledge
rather than absolute, which improves robustness in late-stage refinement.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level deadline guard
_start_time = time.time()
_DEADLINE = _start_time + 300.0  # 300s wall-clock limit per evaluation


def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores for 2000 candidate points.
    
    Parameters:
    -----------
    mu : array of shape (2000,)
        GP posterior mean at candidates (in objective units)
    sigma : array of shape (2000,)
        GP posterior standard deviation at candidates (in objective units)
    f_best : float
        Best (lowest) observed value so far
    
    Returns:
    --------
    scores : array of shape (2000,)
        Acquisition scores; higher means evaluate next
    """
    # Deadline safety check
    if time.time() > _DEADLINE - 5.0:
        # Fallback: Simple UCB with moderate exploration
        sigma_safe = np.maximum(sigma, 1e-9)
        return -mu + 2.0 * sigma_safe

    # Clip sigma to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Define reference values for the Styblinski-Tang function on [-5, 5]
    f_min_ref = -39.166
    f_max_ref = 40.0
    
    # Calculate progress: 0.0 at start (f_best ~ 40), 1.0 at end (f_best ~ -39)
    progress = (f_max_ref - f_best) / (f_max_ref - f_min_ref)
    progress = np.clip(progress, 0.0, 1.0)
    
    # Dynamic exploration coefficient (kappa)
    # Start with high exploration to find the global minimum basin.
    # Decay to low exploration to refine the solution.
    kappa_start = 8.0
    kappa_end = 0.5
    kappa = kappa_end + (kappa_start - kappa_end) * (1.0 - progress) ** 0.5
    
    # Structural Change: Normalize sigma by its median.
    # This makes the exploration term relative to the current global uncertainty.
    # In early stages, sigma is large and similar, so normalization has little effect.
    # In late stages, sigma is small, but relative differences in uncertainty
    # between regions become important for targeted refinement.
    sigma_med = np.median(sigma_safe)
    # Floor the median to prevent division by zero if GP is overconfident
    sigma_scale = max(sigma_med, 1e-6)
    sigma_norm = sigma_safe / sigma_scale
    
    # Base UCB score for minimization: -mu + kappa * sigma_norm
    # -mu encourages exploitation of low predicted values.
    # kappa * sigma_norm encourages exploration of relatively uncertain regions.
    scores = -mu + kappa * sigma_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
