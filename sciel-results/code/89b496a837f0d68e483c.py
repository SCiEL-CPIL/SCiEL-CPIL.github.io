"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best):
    """
    Enhanced Thompson Sampling with adaptive exploration for Bayesian optimization.
    
    Uses per-point posterior samples with an uncertainty-weighted exploration
    bonus that scales with the gap between f_best and the predicted mean,
    encouraging exploration in regions where improvement is possible.
    
    Parameters:
    -----------
    mu : ndarray, shape (n,)
        Mean of the GP posterior at candidate points (in objective units).
    sigma : ndarray, shape (n,)
        Standard deviation of the GP posterior at candidate points (in objective units).
    f_best : float
        The lowest observed objective value so far.
    
    Returns:
    --------
    scores : ndarray, shape (n,)
        Acquisition scores; higher means evaluate next.
    """
    n = len(mu)
    
    # Safety: ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, 1e-12)
    
    # Thompson Sampling: draw per-point standard normal samples
    z = np.random.standard_normal(n)
    
    # Posterior predictive samples: y = mu + sigma * z
    samples = mu + sigma * z
    
    # Compute the gap between f_best and the predicted mean
    # Points with lower mu relative to f_best are more promising for improvement
    gap = f_best - mu  # positive where mu < f_best (potential improvement)
    
    # Normalize the gap to [0, 1] range for scaling
    gap_max = np.max(gap) if np.any(gap > 0) else 1.0
    if gap_max <= 0:
        gap_norm = np.zeros(n)
    else:
        gap_norm = np.clip(gap / gap_max, 0, 1)
    
    # Adaptive exploration bonus: weight by both uncertainty and potential improvement
    # Higher uncertainty + larger gap = more exploration incentive
    sigma_max = np.max(sigma)
    if sigma_max > 1e-9:
        s_norm = sigma / sigma_max
    else:
        s_norm = np.zeros(n)
    
    # Exploration bonus: quadratic in uncertainty, linear in improvement potential
    # This creates a strong preference for uncertain regions with high improvement potential
    exploration_weight = 0.5 * s_norm * (0.5 + 0.5 * gap_norm)
    
    # Combine: negative sample (exploitation) + exploration bonus
    scores = -samples + exploration_weight * sigma_max
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
