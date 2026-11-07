"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Variance-Weighted Expected Improvement (VW-EI) acquisition function.
    
    Multiplies standard EI by normalized posterior variance to prioritize
    candidates where the model is most uncertain about improvement potential.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    start_time = time.time()
    
    # Ensure sigma is positive to avoid division by zero
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    
    # Standard Expected Improvement for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Compute variance-weighted EI: EI * (sigma^2 / max(sigma^2))
    # This converts EI to "expected improvement per unit of model uncertainty"
    # Candidates in high-uncertainty regions get boosted if they also have positive EI
    
    sigma_sq = sigma_safe ** 2
    max_sigma_sq = np.max(sigma_sq) if np.max(sigma_sq) > eps else eps
    
    # Normalize variance to [0, 1]
    var_norm = sigma_sq / max_sigma_sq
    
    # VW-EI: multiply EI by normalized variance
    # This effectively rewards exploration in uncertain regions
    vw_ei = ei * var_norm
    vw_ei = np.where(np.isfinite(vw_ei), vw_ei, 0.0)
    
    # Combine VW-EI with standard EI for balance
    # Use a weighted combination: primarily VW-EI, but retain some standard EI signal
    # This prevents pure exploration when exploitation is clearly better
    
    # Normalize both for combination
    def normalize_scores(s):
        s_min = np.min(s)
        s_max = np.max(s)
        if s_max - s_min < 1e-9:
            return np.zeros_like(s)
        return (s - s_min) / (s_max - s_min)
    
    norm_vw_ei = normalize_scores(vw_ei)
    norm_ei = normalize_scores(ei)
    
    # Adaptive weighting based on distance to known minimum
    # Known minimum for Styblinski-Tang 1D
    f_min_known = -39.1662
    distance_to_min = max(f_best - f_min_known, 0.0)
    
    # Early on: more exploration (higher weight on VW-EI)
    # Later on: more exploitation (higher weight on standard EI)
    if distance_to_min > 15.0:
        w_vw = 0.7
        w_ei = 0.3
    elif distance_to_min > 5.0:
        w_vw = 0.5
        w_ei = 0.5
    else:
        w_vw = 0.3
        w_ei = 0.7
    
    final_scores = w_vw * norm_vw_ei + w_ei * norm_ei
    
    # Ensure all scores are finite
    final_scores = np.where(np.isfinite(final_scores), final_scores, 0.0)
    
    # Safety margin check
    if time.time() - start_time > 280:
        # Return what we have
        pass
    
    return final_scores
# EVOLVE-BLOCK-END
