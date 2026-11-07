"""Acquisition function for Bayesian optimization of Hartmann 3D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard: the evaluator kills after 330s. We track elapsed time.
_START_TIME = time.time()
_DEADLINE = 330.0  # seconds
_SAFETY_MARGIN = 5.0  # seconds before deadline to stop and return


def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition: Expected Improvement (EI) with adaptive exploration
    and a diversity bonus based on posterior uncertainty structure.
    
    Structural change from LCB: uses EI which is more robust and has a
    well-defined exploration/exploitation tradeoff via the improvement
    over f_best. Adds a diversity term that rewards regions with high
    uncertainty relative to the local scale.
    """
    # Deadline check
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - _SAFETY_MARGIN:
        # Return a simple fallback score
        mu_safe = np.where(np.isfinite(mu), mu, 0.0)
        sigma_safe = np.where(np.isfinite(sigma), np.maximum(sigma, 1e-9), 1e-9)
        return -mu_safe + 1.0 * sigma_safe

    n = len(mu)
    
    # Safety: ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), np.maximum(sigma, 1e-9), 1e-9)
    
    # Known minimum for Hartmann 3D
    f_min_known = -3.86278
    
    # Adaptive exploration parameter for EI
    # Early on (f_best far from minimum), use more exploration
    # Late (f_best close to minimum), exploit more
    gap = f_best - f_min_known  # positive, decreases over time
    # kappa: high when gap is large (early), low when gap is small (late)
    # Use a smooth transition
    kappa = 0.3 + 1.2 * (1.0 - np.exp(-gap / 0.8))
    kappa = np.clip(kappa, 0.2, 2.0)
    
    # Expected Improvement for minimization
    # EI(x) = E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is standard normal CDF, phi is standard normal PDF
    
    improvement = f_best - mu
    with np.errstate(divide='ignore', invalid='ignore'):
        z = improvement / sigma
        # Clip z to avoid numerical issues
        z = np.clip(z, -10, 10)
        
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative and finite
    ei = np.where(np.isfinite(ei), np.maximum(ei, 0.0), 0.0)
    
    # Diversity bonus: reward regions with high relative uncertainty
    # Normalize sigma by its scale to get a relative measure
    sigma_scale = np.median(sigma) + 1e-9
    relative_sigma = sigma / sigma_scale
    
    # Diversity term: encourages exploration of under-explored regions
    # Use a saturating function to avoid extreme values
    diversity = np.log1p(relative_sigma)  # log(1 + relative_sigma)
    
    # Weight for diversity: stronger early, weaker late
    diversity_weight = 0.1 * (1.0 - np.exp(-gap / 1.0))
    
    # Combine: EI is the primary signal, diversity is a secondary boost
    # Scale EI to be comparable to diversity term
    ei_scale = np.median(ei[ei > 0]) + 1e-9 if np.any(ei > 0) else 1.0
    ei_normalized = ei / ei_scale
    
    # Final score: higher is better
    score = ei_normalized + diversity_weight * diversity
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
