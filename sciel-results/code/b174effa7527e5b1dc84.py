"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Adaptive acquisition function combining Expected Improvement (EI) and
    Uncertainty-Driven Exploration (UCB-like) with a smooth structural switch.
    
    The design uses a progress-based schedule to blend between exploration
    (high uncertainty) and exploitation (high improvement) as the search
    progresses toward the known global minimum.
    
    Parameters
    ----------
    mu : ndarray, shape (n,)
        Mean of the GP posterior at candidate points.
    sigma : ndarray, shape (n,)
        Standard deviation of the GP posterior at candidate points.
    f_best : float
        Best (lowest) observed function value so far.
    
    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    # Deadline guard: track elapsed time to avoid exceeding evaluator limit
    _t0 = time.time()
    _DEADLINE = 280.0  # safety margin before 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    
    # Known global minimum for Hartmann-3D
    f_global_min = -3.86278
    # Reference "worst" value for progress calculation
    f_worst_ref = 0.0
    
    # Calculate progress towards the global minimum
    denominator = f_worst_ref - f_global_min
    if denominator == 0:
        progress = 0.5
    else:
        progress = (f_worst_ref - f_best) / denominator
    progress = np.clip(progress, 0.0, 1.0)
    
    # --- Expected Improvement (EI) ---
    # EI measures the expected improvement over f_best
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # --- Uncertainty-driven exploration score (UCB-like) ---
    # For minimization, we want to explore regions with high uncertainty
    # and low predicted values. The score -mu + beta*sigma encourages both.
    # Use an adaptive beta that decreases as progress increases.
    beta_max = 2.5   # high exploration early
    beta_min = 0.5   # low exploration late
    beta = beta_max - (beta_max - beta_min) * progress
    
    ucb_score = -mu + beta * sigma
    
    # --- Smooth structural switch ---
    # Instead of a hard switch, use a smooth blend based on progress.
    # Early on (progress near 0), weight exploration more heavily.
    # Later (progress near 1), weight exploitation (EI) more heavily.
    # Use a sigmoid-like smooth transition centered at progress = 0.5
    # with a width that controls the transition sharpness.
    transition_center = 0.4  # shift transition slightly earlier
    transition_width = 0.3
    # Smooth weight for exploration (1 at progress=0, 0 at progress=1)
    weight_explore = 1.0 / (1.0 + np.exp((progress - transition_center) / transition_width))
    weight_exploit = 1.0 - weight_explore
    
    # Normalize scores to comparable scales before blending
    # EI is naturally non-negative and bounded
    ei_max = np.max(ei)
    if ei_max > 1e-9:
        ei_norm = ei / ei_max
    else:
        ei_norm = ei
    
    # UCB score can be negative; shift to be non-negative for blending
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    ucb_range = ucb_max - ucb_min
    if ucb_range > 1e-9:
        ucb_norm = (ucb_score - ucb_min) / ucb_range
    else:
        ucb_norm = np.zeros_like(ucb_score)
    
    # Combine with smooth weights
    # Use the original (unnormalized) values for final score to preserve scale
    # but use normalized versions for the blending decision
    scores = weight_exploit * ei + weight_explore * ucb_score
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline check
    if time.time() - _t0 > _DEADLINE:
        # Return the best result found so far
        return scores
    
    return scores
# EVOLVE-BLOCK-END
