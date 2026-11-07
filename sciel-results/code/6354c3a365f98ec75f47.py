"""Rewritten acquisition for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function that blends Expected Improvement (exploitation)
    with Upper Confidence Bound (exploration) for Bayesian optimization.
    
    The blend weight is adaptive: early on (high uncertainty relative to
    improvement potential) we favor exploration; later we shift toward
    exploitation to refine the best candidate.
    """
    # Ensure sigma is positive for numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Compute UCB-style exploration score
    # Use a beta that balances exploration and exploitation
    # We normalize sigma to [0, 1] for a stable blend
    sigma_max = np.max(sigma_safe)
    sigma_min = np.min(sigma_safe)
    
    if sigma_max > sigma_min + 1e-12:
        sigma_norm = (sigma_safe - sigma_min) / (sigma_max - sigma_min)
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Adaptive beta: start with higher exploration, decrease as we converge
    # Measure how close the best prediction is to f_best
    mu_min = np.min(mu)
    improvement_potential = f_best - mu_min  # How much improvement is possible
    
    # If improvement potential is large relative to uncertainty, explore more
    # If improvement potential is small, exploit more
    sigma_mean = np.mean(sigma_safe)
    if sigma_mean > 1e-10:
        ratio = improvement_potential / (sigma_mean + 1e-10)
    else:
        ratio = 0.0
    
    # Beta decays from ~2.0 (high exploration) to ~0.5 (high exploitation)
    # based on the ratio of improvement potential to uncertainty
    beta = 0.5 + 1.5 * np.exp(-abs(ratio) / 5.0)
    
    # Combine EI and exploration
    # EI is in objective units, sigma_norm is in [0,1]
    # Scale sigma_norm to be comparable to EI
    ei_max = np.max(ei)
    if ei_max > 1e-10:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    # Final score: weighted combination
    # Higher score means more promising
    score = (1.0 - beta / 3.0) * ei_norm + (beta / 3.0) * sigma_norm
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
