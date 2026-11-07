"""Cost-aware acquisition for BO on shifted Hartmann 3D (5 seeds, total cost < 30)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Global deadline guard: kill long-running computations safely.
_T0 = time.time()
_HARD_DEADLINE = 320.0  # seconds; leave ~10s buffer for the harness


def _elapsed_ok():
    return (time.time() - _T0) < _HARD_DEADLINE


def _expected_improvement(mu, sigma, f_best):
    # Standard EI (for minimization): (f_best - mu) * Phi(z) + sigma * phi(z)
    # Guard against sigma == 0
    sigma_safe = np.where(sigma > 1e-12, sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # Where sigma is effectively zero, EI is just max(0, f_best - mu)
    ei = np.where(sigma > 1e-12, ei, np.maximum(0.0, f_best - mu))
    return ei


def acquisition(mu, sigma, f_best, cost):
    """
    Return 2000 finite scores (higher = better to evaluate next).
    
    Strategy: Probability of Improvement (POI) per unit cost.
    POI is more robust than EI in low-budget scenarios where EI values can be
    very small and dominated by noise. Dividing by cost ensures we prefer
    cheaper evaluations when the probability of improvement is similar.
    """
    if not _elapsed_ok():
        return np.full_like(mu, 0.0)
    
    sigma_safe = np.where(sigma > 1e-12, sigma, 1e-12)
    cost_safe = np.where(cost > 1e-9, cost, 1e-9)
    
    # Probability of Improvement: P(f(x) < f_best)
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)
    
    # Where sigma is effectively zero, POI is 1 if mu < f_best, else 0
    poi = np.where(sigma > 1e-12, poi, np.where(mu < f_best, 1.0, 0.0))
    
    # Cost-aware POI
    scores = poi / cost_safe
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
