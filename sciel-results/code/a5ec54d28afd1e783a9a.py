"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: the evaluator kills after 330s; we aim to finish acquisition
# well within per-call limits. We track a soft deadline for the whole process.
# However, since `acquisition` is called repeatedly by the evaluator's loop,
# we cannot easily track global elapsed time across calls unless we use a
# module-level variable. Let's use a simple heuristic: if the computation
# seems too expensive, we could bail, but with 2000 points it should be fast.
# The main cost is the norm.cdf/pdf which are vectorized.

def acquisition(mu, sigma, f_best, cost):
    # Ensure arrays are float64 for precision
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF
    # Using scipy.stats.norm for stability
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: if mu > f_best, EI can be negative. 
    # We can clip EI to be non-negative, or just let it be.
    # Usually, acquisition functions are clipped to 0 if no improvement is expected.
    # However, for cost-awareness, we divide by cost.
    # If EI is negative, the score is negative.
    # Let's calculate the raw EI first.
    ei = (f_best - mu) * cdf + sigma_safe * pdf
    
    # Clip EI to be non-negative? 
    # If mu is already better than f_best, EI is negative. 
    # In minimization, we want to find points with f(x) < f_best.
    # If mu < f_best, we expect improvement.
    # If mu > f_best, we don't.
    # So EI should be max(0, ...).
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware score: EI / cost
    score = ei / cost_safe
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
