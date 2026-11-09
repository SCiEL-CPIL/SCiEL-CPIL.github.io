"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import erf

_DEADLINE_START = time.time()
_TIME_LIMIT = 295.0  # seconds, safety margin before 330s kill


def _norm_cdf(x):
    """Fast standard normal CDF using erf."""
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def _norm_pdf(x):
    """Fast standard normal PDF."""
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Expected Improvement (EI) for minimization.
    
    EI(x) = (f_best - mu) * Phi(-d) + sigma * phi(-d)
    where d = (mu - f_best) / sigma
    
    We divide by cost to make it cost-aware.
    """
    # Check time deadline
    if time.time() - _DEADLINE_START > _TIME_LIMIT:
        # Fallback: simple EI / cost
        sigma_safe = np.maximum(sigma, 1e-8)
        d = (mu - f_best) / sigma_safe
        ei = (f_best - mu) * _norm_cdf(-d) + sigma * _norm_pdf(-d)
        cost_safe = np.maximum(cost, 1e-9)
        score = ei / cost_safe
        return np.where(np.isfinite(score), score, 0.0)

    # Compute Expected Improvement for minimization
    sigma_safe = np.maximum(sigma, 1e-8)
    d = (mu - f_best) / sigma_safe
    
    # EI = (f_best - mu) * Phi(-d) + sigma * phi(-d)
    # When mu < f_best, d < 0, so -d > 0, Phi(-d) > 0.5
    # When mu > f_best, d > 0, so -d < 0, Phi(-d) < 0.5
    ei = (f_best - mu) * _norm_cdf(-d) + sigma * _norm_pdf(-d)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost to penalize expensive evaluations
    cost_safe = np.maximum(cost, 1e-9)
    score = ei / cost_safe
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
