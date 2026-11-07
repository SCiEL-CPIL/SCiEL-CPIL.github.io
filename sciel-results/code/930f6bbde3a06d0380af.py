"""Cost-aware acquisition: Cost-Effective Expected Improvement (CEI)."""
import numpy as np
from scipy.stats import norm
import time

# EVOLVE-BLOCK-START

def _safe_ei(mu, sigma, f_best):
    """
    Compute Expected Improvement for minimization.
    EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    where z = (f_best - mu) / sigma
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues in norm.cdf/pdf for extreme values
    z = np.clip(z, -10, 10)
    
    # Standard normal CDF and PDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # EI formula
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # EI should be non-negative
    return np.maximum(ei, 0.0)

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Cost-Effective Expected Improvement.
    
    Score = EI(x) / cost(x)
    
    This directly optimizes the expected improvement per unit of cost.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety checks
    # Cost is 1 + mean(x scaled to [0,1]), so min cost is 1.
    # However, to be safe against any edge cases or future changes, we floor it.
    cost_safe = np.maximum(cost, 1e-6)
    
    # Calculate Expected Improvement
    ei = _safe_ei(mu, sigma, f_best)
    
    # Cost-Effective Score
    # We divide by cost to prefer cheaper points that offer similar improvement.
    score = ei / cost_safe
    
    # Optional: Add a small exploration bonus for high uncertainty, also cost-normalized.
    # This helps if EI is 0 everywhere (e.g., f_best is very low and mu > f_best everywhere).
    # However, pure EI/cost is robust. Let's stick to the high-performing inspiration strategy.
    # If we want to be slightly more aggressive in exploration early on, we could add
    # kappa * sigma / cost. But without a clear phase signal, static kappa is risky.
    # The inspiration 1 (1.337) suggests EI/cost is strong.
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score

# EVOLVE-BLOCK-END
