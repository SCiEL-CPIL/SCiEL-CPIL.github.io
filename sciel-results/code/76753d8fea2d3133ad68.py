"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Global deadline guard: keep a safety margin before the 300 s kill.
_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0  # 20 s margin


def _time_guard():
    """Return True if we are within the safe time budget."""
    return time.time() < _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI).
    Returns 2000 finite scores; higher = evaluate next.
    """
    # Deadline guard: if running out of time, return a simple valid fallback.
    if not _time_guard():
        # Fallback: favor cheap, uncertain points
        c = np.maximum(cost, 1e-9)
        s = np.maximum(sigma, 1e-9)
        return s / c

    eps = 1e-12
    s = np.maximum(sigma, eps)
    c = np.maximum(cost, eps)

    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / s
    # Clip z for numerical stability
    z = np.clip(z, -10.0, 10.0)
    
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    ei = (f_best - mu) * Phi_z + s * phi_z
    ei = np.maximum(ei, 0.0)  # EI should be non-negative

    # Cost-aware score: EI per unit cost
    score = ei / c

    # Add a small exploration bonus proportional to uncertainty/cost
    # This helps escape local optima and encourages diverse search
    kappa = 0.1
    score_explore = kappa * s / c

    # Combine
    score = score + score_explore

    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    return score
# EVOLVE-BLOCK-END
