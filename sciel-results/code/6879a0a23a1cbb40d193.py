"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    
    Uses a combination of Lower Confidence Bound (LCB) and Expected Improvement (EI)
    with an exploration bonus to balance exploitation and exploration.
    
    The LCB approach: LCB = mu - kappa * sigma
    where kappa controls the exploration-exploitation tradeoff.
    
    We use a dynamic kappa that adapts based on the uncertainty relative to f_best,
    and add a small EI component to ensure we still have a well-defined improvement
    signal.
    """
    # Handle numerical stability for sigma close to 0
    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    
    # Dynamic exploration parameter:
    # When f_best is large (early in optimization), we want more exploration.
    # When f_best is small (close to optimum), we want more exploitation.
    # Scale kappa based on the magnitude of f_best and the uncertainty.
    f_best_abs = abs(f_best)
    sigma_mean = np.mean(sigma_safe)
    
    # kappa decreases as we get closer to the optimum
    # Use a soft scaling: kappa ~ 1 / (1 + f_best_abs / sigma_mean)
    if sigma_mean > eps:
        kappa = 1.0 / (1.0 + f_best_abs / sigma_mean)
    else:
        kappa = 1.0
    
    # Clip kappa to a reasonable range
    kappa = np.clip(kappa, 0.5, 2.0)
    
    # LCB score: lower is better, so we negate for "higher is better"
    lcb_score = -(mu - kappa * sigma_safe)
    
    # Also compute EI for a combined score
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    # Combine: primarily LCB, with a small EI component to break ties
    # and ensure we prefer points that have expected improvement
    score = lcb_score + 0.1 * ei
    
    # Ensure scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
