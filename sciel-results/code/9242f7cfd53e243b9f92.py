"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Per-process deadline guard
_T0 = time.time()
_DEADLINE = 330.0


def _elapsed():
    return time.time() - _T0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses Expected Improvement per unit cost as the primary signal,
    with a deadline guard for safety.
    """
    # Guard: if we are dangerously close to the wall-clock limit, return a
    # deterministic, finite, low-risk signal immediately.
    if _elapsed() > _DEADLINE - 1.0:
        return np.zeros(np.shape(mu), dtype=float)

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    # Sanitize inputs
    sigma = np.clip(sigma, 1e-9, None)
    cost = np.clip(cost, 1e-9, None)
    f_best = float(f_best) if np.isfinite(f_best) else 0.0

    # Calculate Expected Improvement (for minimization)
    # EI(x) = E[max(f_best - f(x), 0)]
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    
    z = (f_best - mu) / sigma
    # Clip z to avoid extreme values that might cause numerical issues
    z = np.clip(z, -10.0, 10.0)
    
    # Compute EI
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Exploration term: UCB-like, scaled by cost
    # beta controls the trade-off between exploitation (EI) and exploration (sigma)
    beta = 1.0
    ucb = beta * sigma
    
    # Cost-aware score: (EI + UCB) / cost
    # We want to maximize this score.
    score = (ei + ucb) / cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
