"""Acquisition function for Bayesian optimization of a fixed seeded random RBF mixture in 2 dimensions."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Local Intensive Expected Improvement (LIEI) acquisition function.
    
    Clips the improvement term (f_best - mu) to be non-negative, suppressing
    exploration in regions where the GP predicts values worse than the current best.
    """
    # Standard EI components
    z = (f_best - mu) / (sigma + 1e-9)
    
    # Clip the improvement to be non-negative: only consider points where mu < f_best
    improvement = np.maximum(f_best - mu, 0.0)
    
    # Standard EI formula but with clipped improvement
    # EI = improvement * Phi(z) + sigma * phi(z)
    # But we want to suppress exploration where mu >= f_best (improvement = 0)
    # In those regions, z <= 0, so Phi(z) <= 0.5 and phi(z) is the pdf
    
    # LIEI: multiply by a factor that strongly damps regions where mu >= f_best
    # Use a smooth gating function based on z
    # When z is very negative (mu >> f_best), we want score ~ 0
    # When z is positive (mu < f_best), we want the full EI
    
    # Compute standard EI
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Apply a gating factor: sigmoid-like based on z
    # gate = 1 / (1 + exp(-alpha * z)) where alpha controls sharpness
    # For large positive z, gate -> 1; for large negative z, gate -> 0
    alpha = 2.0
    gate = 1.0 / (1.0 + np.exp(-alpha * z))
    
    # The gated EI
    scores = ei * gate
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
