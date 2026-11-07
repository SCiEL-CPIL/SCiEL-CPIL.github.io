"""Cost-aware acquisition function for Bayesian optimization (powell 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Ratio-based Expected Improvement.
    
    Strategy: Compute Expected Improvement (EI) for minimization and divide by cost.
    This creates a 'value per cost unit' metric that naturally balances exploration
    (high sigma -> high EI) and cost efficiency. As f_best improves, the potential
    EI decreases, shifting focus to exploitation without needing explicit beta tuning.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Calculate Z-score for EI
    # Z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(-z) + sigma * phi(z)
    # Note: Phi is CDF, phi is PDF
    phi_z = norm.pdf(z)
    phi_neg_z = norm.cdf(-z)
    
    ei = (f_best - mu) * phi_neg_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Structural Change: Ratio-based score
    # Score = EI / Cost
    # This explicitly optimizes for improvement per unit cost.
    # High cost points are heavily penalized if they don't offer proportional EI.
    score = ei / cost_safe
    
    # Optional: Add a small exploration bonus based on sigma to avoid getting stuck
    # in local minima if EI becomes very small but sigma is still non-zero?
    # The ratio EI/cost already captures this. If sigma is high, EI is high.
    # However, if f_best is very close to mu, EI is small.
    # To ensure we don't return all zeros if we are stuck, we can add a tiny
    # fraction of sigma/cost?
    # Let's stick to pure EI/cost first as it's a cleaner structural change.
    
    # Ensure all outputs are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
