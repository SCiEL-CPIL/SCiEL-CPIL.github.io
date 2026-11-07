"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard
_start_time = time.time()
_DEADLINE = 280.0  # Safety margin before 300s limit


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Goldstein-Price 2D.
    
    Uses a hybrid approach:
    - Early stage: aggressive exploration via LCB with high beta
    - Later stage: exploitation via Expected Improvement (EI)
    - Transition based on the ratio of f_best to the known minimum (3.0)
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviations
    f_best : float - best observed value so far
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = better)
    """
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        # Emergency: return something finite
        return np.zeros_like(mu)
    
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Determine the stage based on how close we are to the optimum
    # Goldstein-Price 2D minimum is 3.0
    # If f_best is much larger than 3.0, we're in early exploration
    # If f_best is close to 3.0, we're in exploitation
    
    # Progress metric: how much have we improved from a typical initial value?
    # Initial uniform random points on [-2,2]^2 for Goldstein-Price typically give values in [3, 30]
    # Let's use a smooth transition
    
    # Use a logistic-like transition based on f_best
    # When f_best is large (e.g., > 15), we want high exploration
    # When f_best is close to 3.0, we want exploitation
    
    # Define transition thresholds
    f_explore = 10.0  # Above this, fully explore
    f_exploit = 5.0   # Below this, fully exploit
    
    if f_best >= f_explore:
        exploration_weight = 1.0
    elif f_best <= f_exploit:
        exploration_weight = 0.0
    else:
        # Smooth transition
        exploration_weight = (f_best - f_exploit) / (f_explore - f_exploit)
    
    # LCB score: maximize -mu + beta * sigma
    # Use a beta that balances exploration and exploitation
    beta_lcb = 2.0  # Aggressive exploration
    
    lcb_scores = -mu + beta_lcb * sigma_safe
    
    # EI score for minimization
    z = (f_best - mu) / sigma_safe
    ei_scores = (f_best - mu) * 0.5 * (1 + _norm_cdf(z)) + sigma_safe * _norm_pdf(z)
    
    # Combine: blend between LCB and EI based on exploration_weight
    # We need to normalize them to be comparable
    # Use a simple weighted average after scaling
    
    # Scale both to have similar magnitudes for blending
    lcb_mean = np.mean(lcb_scores)
    lcb_std = np.std(lcb_scores) + eps
    lcb_normalized = (lcb_scores - lcb_mean) / lcb_std
    
    ei_mean = np.mean(ei_scores)
    ei_std = np.std(ei_scores) + eps
    ei_normalized = (ei_scores - ei_mean) / ei_std
    
    # Blend
    scores = exploration_weight * lcb_normalized + (1 - exploration_weight) * ei_normalized
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores


def _norm_cdf(x):
    """Standard normal CDF using erf."""
    return 0.5 * (1 + _erf(x / np.sqrt(2)))


def _norm_pdf(x):
    """Standard normal PDF."""
    return np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)


def _erf(x):
    """Error function approximation."""
    # Use numpy's erf if available, otherwise approximate
    try:
        from scipy.special import erf
        return erf(x)
    except ImportError:
        # Abramowitz and Stegun approximation
        sign = np.sign(x)
        x = np.abs(x)
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
        return sign * y
# EVOLVE-BLOCK-END
