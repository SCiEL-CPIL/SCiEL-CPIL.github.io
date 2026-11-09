"""Acquisition function for Bayesian optimization of Weierstrass 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: combination of Expected Improvement (EI) and
    Lower Confidence Bound (LCB) with adaptive exploration weight.
    
    Parameters
    ----------
    mu : ndarray, shape (2000,)
        Posterior mean at candidate points.
    sigma : ndarray, shape (2000,)
        Posterior standard deviation at candidate points.
    f_best : float
        Best (lowest) observed function value.
    
    Returns
    -------
    scores : ndarray, shape (2000,)
        Acquisition values; higher means evaluate next.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement for minimization
    improvement = f_best - mu
    z = improvement / sigma_safe
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Lower Confidence Bound (negative so higher = more promising)
    # Use beta that balances exploration and exploitation
    beta = 2.0
    lcb = -(mu + beta * sigma_safe)
    
    # Normalize both components to have similar scale
    ei_max = np.max(ei)
    if ei_max > 1e-9:
        ei_norm = ei / ei_max
    else:
        ei_norm = ei
    
    lcb_min = np.min(lcb)
    lcb_max = np.max(lcb)
    lcb_range = lcb_max - lcb_min
    if lcb_range > 1e-9:
        lcb_norm = (lcb - lcb_min) / lcb_range
    else:
        lcb_norm = lcb - lcb_min
    
    # Adaptive weight: more exploration early (when sigma is large relative to improvement)
    # Measure how much uncertainty remains
    sigma_mean = np.mean(sigma_safe)
    improvement_mean = np.mean(np.abs(improvement))
    
    if improvement_mean > 1e-9:
        uncertainty_ratio = sigma_mean / improvement_mean
    else:
        uncertainty_ratio = 1.0
    
    # Weight for LCB: increase when uncertainty is high
    # Clip to keep in reasonable range
    uncertainty_ratio = np.clip(uncertainty_ratio, 0.1, 5.0)
    w_lcb = 0.3 + 0.4 * min(uncertainty_ratio, 2.0) / 2.0
    w_ei = 1.0 - w_lcb
    
    # Combine
    scores = w_ei * ei_norm + w_lcb * lcb_norm
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
