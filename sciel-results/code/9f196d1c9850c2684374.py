"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Deterministic cost-penalized Expected Improvement (EI) acquisition function.
    
    Strategy:
    Use standard EI for robustness on multimodal functions like Hartmann 3D.
    Apply a linear penalty (lambda * cost) instead of division or multiplicative
    factors to avoid over-biasing against expensive regions where the minimum
    might be located. This structural change (subtraction vs division) provides
    a more balanced trade-off between finding improvement and minimizing cost.
    """
    # Deadline guard: if we're getting close to timeout, return a safe fallback
    elapsed = time.time() - _start_time
    if elapsed > 290:  # Safety margin before 300s limit (evaluator kills at 330s)
        # Fallback: simple greedy exploitation with light cost penalty
        # This should be fast and finite
        sigma_safe = np.maximum(sigma, 1e-8)
        z = (f_best - mu) / sigma_safe
        z_clipped = np.clip(z, -30, 30)
        ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
        ei = np.maximum(ei, 0.0)
        scores = ei - 0.1 * cost
        return np.where(np.isfinite(scores), scores, 0.0)

    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate z-score for EI
    # z = (f_best - mu) / sigma
    # Since we are minimizing, improvement is f_best - mu (positive if mu < f_best)
    diff = f_best - mu
    z = diff / sigma_safe
    
    # Clip z to prevent numerical issues in CDF/PDF
    z_clipped = np.clip(z, -30, 30)
    
    # Calculate Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    phi_z = norm.pdf(z_clipped)
    Phi_z = norm.cdf(z_clipped)
    
    ei = diff * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (numerical artifacts might make it slightly negative)
    ei = np.maximum(ei, 0.0)
    
    # Deterministic cost penalty: subtract lambda * cost
    # lambda = 0.1 is small enough to not dominate the EI signal
    # but large enough to prefer cheaper points when EI is similar
    lambda_cost = 0.1
    scores = ei - lambda_cost * cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
