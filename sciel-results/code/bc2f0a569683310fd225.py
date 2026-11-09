# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: the evaluator kills after 330s wall-clock.
# We track time since first call to stay within budget.
_t_start = None

def acquisition(mu, sigma, f_best, cost):
    global _t_start
    if _t_start is None:
        _t_start = time.time()
    
    # Safety: if we're running out of time, return a simple but valid score
    elapsed = time.time() - _t_start
    if elapsed > 310.0:
        # Return a valid finite array immediately using ER/cost
        eps = 1e-9
        sigma_safe = np.maximum(sigma, eps)
        z = (f_best - mu) / sigma_safe
        cdf_z = norm.cdf(z)
        pdf_z = norm.pdf(z)
        er = (f_best - mu) * cdf_z + sigma * pdf_z
        er = np.maximum(er, 0.0)
        score = er / (cost + eps)
        score = np.where(np.isfinite(score), score, 0.0)
        return score
    
    # Cost-aware Expected Regret (ER) acquisition.
    # ER = E[max(0, f_best - f(x))] = (f_best - mu)*Phi(z) + sigma*phi(z)
    # where z = (f_best - mu) / sigma.
    # This captures both the probability AND magnitude of improvement,
    # which is critical for Rastrigin's multi-modal landscape where
    # large jumps between basins are more valuable than small improvements.
    # Dividing by cost creates a trade-off: high expected regret AND low cost.
    
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    
    # Expected Regret: E[max(0, f_best - f(x))]
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    er = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Ensure non-negative (numerical safety)
    er = np.maximum(er, 0.0)
    
    # Cost-aware score: expected regret per unit cost
    score = er / (cost + eps)
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
