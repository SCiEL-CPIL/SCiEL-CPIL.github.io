"""Rewritten acquisition for Bayesian optimization of sphere 1d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Expected Improvement (EI) with adaptive scaling
    for Bayesian optimization of sphere 1d (minimization).
    
    EI is the standard, robust choice for smooth functions. We add adaptive
    scaling to handle numerical stability across different optimization stages.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    fb = float(f_best)
    
    # Ensure sigma is strictly positive
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Standardized improvement
    z = (fb - mu) / sigma_safe
    
    # Expected Improvement for minimization
    # EI = (fb - mu) * Phi(z) + sigma * phi(z)
    ei = (fb - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Adaptive scaling: normalize EI by the typical scale of improvement
    # This helps balance exploration/exploitation across different stages
    # When fb is large (early), improvements are large; when fb is small (late),
    # improvements are tiny. Normalizing keeps the acquisition well-scaled.
    
    # Use the median absolute deviation of mu as a scale reference
    # This is more robust than std
    mu_scale = np.median(np.abs(mu - np.median(mu)))
    if mu_scale < 1e-10:
        mu_scale = 1.0
    
    # Scale EI relative to the problem scale
    # This prevents EI from becoming numerically tiny when close to optimum
    ei_scaled = ei / (mu_scale + 1e-10)
    
    # Add a small exploration bonus that encourages looking at uncertain regions
    # but only when we're not already very close to the optimum
    # The bonus decays as f_best approaches 0 (the true minimum)
    exploration_weight = np.exp(-np.abs(fb) * 2.0)  # decays as fb -> 0
    exploration_bonus = exploration_weight * sigma_safe / (mu_scale + 1e-10)
    
    # Combine: primary is scaled EI, secondary is exploration bonus
    score = ei_scaled + 0.1 * exploration_bonus
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
