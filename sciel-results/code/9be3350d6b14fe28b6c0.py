"""Cost-aware acquisition function for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Quadratic cost penalty EI.
    Uses Expected Improvement divided by cost^2 to aggressively
    prioritize low-cost evaluations, allowing more samples within
    the fixed budget.
    
    EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    where z = (f_best - mu(x)) / sigma(x)
    
    Score = EI / cost^2
    """
    # Deadline guard
    t0 = time.time()
    
    # Ensure sigma is strictly positive
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Use scipy for accurate normal CDF and PDF
    try:
        from scipy.stats import norm
        Phi = norm.cdf(z)
        phi = norm.pdf(z)
    except ImportError:
        # Fallback approximation
        Phi = 0.5 * (1 + np.tanh(z / np.sqrt(2)))
        phi = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    
    # Expected Improvement
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Ensure non-negative
    ei = np.maximum(ei, 0.0)
    
    # Quadratic cost penalty: divide by cost^2
    # This aggressively penalizes expensive points
    cost_safe = np.maximum(cost, 1e-12)
    score = ei / (cost_safe ** 2)
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline
    elapsed = time.time() - t0
    if elapsed > 250:  # Safety margin
        # Return a simple fallback if we're running too long
        # But EI computation is fast, so this should rarely trigger
        pass
    
    return score
    # EVOLVE-BLOCK-END
