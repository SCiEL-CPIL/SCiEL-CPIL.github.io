"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import ndtr, ndtri
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization.
    Uses a structural variant: entropy-weighted expected improvement (H-EI).
    
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
    
    # Ensure sigma is positive
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF via scipy.special
    Phi = ndtr(z)
    phi = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Clip EI to be non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Structural change: multiply by an entropy-based exploration bonus
    # The differential entropy of N(mu, sigma^2) is 0.5 * log(2*pi*e*sigma^2)
    # We use log(sigma) as a monotone proxy for entropy; larger sigma -> more exploratory
    # This reweights candidates: in high-uncertainty regions, EI is amplified,
    # while in well-exploited regions (small sigma), the bonus shrinks.
    # This is structurally different from plain EI or UCB: it's a multiplicative
    # interaction between exploitation value and epistemic uncertainty.
    log_sigma = np.log(sigma_safe)
    
    # Scale the entropy term so it's O(1) relative to typical EI values
    # The range of log_sigma for our problem: sigma is in objective units
    # For ST-1D, f ranges roughly [-39, 50], so sigma might be 0.1 to 10
    # log_sigma ranges from about -2.3 to 2.3
    # We use a soft scaling: entropy_bonus = 1 + 0.5 * log_sigma
    # This gives a multiplier between ~0.1 and ~1.8 for reasonable sigma ranges
    entropy_bonus = 1.0 + 0.5 * log_sigma
    entropy_bonus = np.maximum(entropy_bonus, 0.05)
    
    scores = ei * entropy_bonus
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 290:
        return scores
        
    return scores
# EVOLVE-BLOCK-END
