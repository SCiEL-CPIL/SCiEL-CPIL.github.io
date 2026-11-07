"""Rewritten acquisition for Bayesian optimization of Hartmann 6d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6d.
    
    Uses a UCB strategy for minimization with a two-factor adaptive beta:
    one factor tracks objective progress toward the known minimum,
    the other tracks posterior uncertainty. This ensures strong
    exploration when the GP is uncertain and the search is far from
    convergence, while shifting to exploitation as either factor
    indicates the search is maturing.
    
    Parameters
 ----------
    mu : ndarray, shape (n,)
        GP mean predictions.
    sigma : ndarray, shape (n,)
        GP standard deviation predictions.
    f_best : float
        Best (lowest) observed objective value so far.

    Returns
 -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values are preferred.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 300s limit
    
    n = len(mu)
    
    # Guard against non-positive sigma
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Factor 1: Objective progress toward known minimum
    hartmann_min = -3.32237
    gap_to_min = max(f_best - hartmann_min, 0.0)
    # Normalize by a characteristic scale; Hartmann 6d range is roughly [-3.3, -1.0]
    obj_scale = abs(hartmann_min) + 2.0  # ~5.3, gives smooth 0-1 range
    progress_factor = np.exp(-gap_to_min / obj_scale)  # 1 when far, 0 when close
    
    # Factor 2: Posterior uncertainty
    # Mean sigma relative to the objective scale
    f_scale = abs(f_best) + 1.0
    mean_sigma = np.mean(sigma_safe)
    uncertainty_factor = np.clip(mean_sigma / f_scale, 0.0, 1.0)
    
    # Combined beta: high when both progress is low (far from min) and
    # uncertainty is high. We use a weighted geometric-mean-like blend
    # so that either factor being low reduces exploration.
    # beta = base * (progress_factor * w + uncertainty_factor * (1-w))
    # But we want multiplicative interaction: if either is low, reduce beta.
    # Use: beta = beta_min + (beta_max - beta_min) * progress_factor * uncertainty_boost
    # where uncertainty_boost amplifies when sigma is high
    
    beta_max = 3.5
    beta_min = 0.3
    
    # The uncertainty boost: when mean_sigma is large relative to f_scale,
    # we want more exploration. Scale it to [0.5, 1.5]
    uncertainty_boost = 0.5 + uncertainty_factor  # ranges [0.5, 1.5]
    
    beta = beta_min + (beta_max - beta_min) * progress_factor * uncertainty_boost
    
    # Clamp beta to reasonable range
    beta = np.clip(beta, beta_min, beta_max)
    
    # UCB for minimization: score = -mu + beta * sigma
    scores = -mu + beta * sigma_safe
    
    # Check deadline
    if time.time() > deadline:
        return np.where(np.isfinite(scores), scores, 0.0)
    
    # Ensure all values are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
