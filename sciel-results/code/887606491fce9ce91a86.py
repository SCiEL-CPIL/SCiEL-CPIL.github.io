"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track elapsed time for deadline guard
_start_time = time.time()
_DEADLINE = 325.0  # 330s limit, leave 5s margin

def _check_deadline():
    """Return True if we should abort and return a safe fallback."""
    return (time.time() - _start_time) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: if we're running out of time, return a safe fallback
    if _check_deadline():
        # Fallback: just return 1/cost to at least prefer cheaper points
        cost_safe = np.maximum(cost, 1e-6)
        return 1.0 / cost_safe

    # Ensure inputs are finite and handle edge cases
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Exploration term: uncertainty-driven
    # Use a moderate exploration weight
    beta = 0.5
    exploration = beta * sigma_safe
    
    # Combined score before cost adjustment
    raw_score = ei + exploration
    
    # Sublinear cost penalty: divide by cost^0.5
    # This allows some exploration in expensive regions while preferring cheaper points
    cost_penalty = np.sqrt(cost_safe)
    score = raw_score / cost_penalty
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
