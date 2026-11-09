"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard: 330s wall-clock limit, use 300s as safety margin
_DEADLINE_START = time.time()
_TIME_LIMIT = 300.0  # seconds, leave 30s margin before 330s kill


def _norm_cdf(x):
    """Numerically stable normal CDF."""
    return 0.5 * (1.0 + np.sign(x) * np.sqrt(1.0 - np.exp(-2.0 * x**2)))


def _norm_pdf(x):
    """Standard normal PDF."""
    return np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using pure Expected Improvement (EI)
    normalized by evaluation cost.
    
    Structural change: remove heuristic exploration bonus and adaptive weights.
    Use the mathematically clean EI/cost ratio, which for single-step selection
    is equivalent to Knowledge Gradient. The cost normalization naturally
    balances exploitation (high EI) with exploration (high sigma leads to
    higher EI in uncertain regions).
    """
    # Check time deadline
    if time.time() - _DEADLINE_START > _TIME_LIMIT:
        # Safe fallback: simple cost-normalized EI
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * _norm_cdf(z) + sigma_safe * _norm_pdf(z)
        ei = np.maximum(ei, 0.0)
        score = ei / cost_safe
        score = np.where(np.isfinite(score), score, 0.0)
        return score
    
    # Safe sigma and cost
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement: E[max(0, f_best - f(x))]
    ei = (f_best - mu) * _norm_cdf(z) + sigma_safe * _norm_pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-normalized EI: prefer cheap evaluations with high expected improvement
    score = ei / cost_safe
    
    # Ensure finite and non-negative
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
