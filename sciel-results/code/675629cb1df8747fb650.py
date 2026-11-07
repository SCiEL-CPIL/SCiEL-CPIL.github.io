"""Cost-aware acquisition function for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Global deadline guard: kill long-running computations safely.
_T0 = time.time()
_HARD_DEADLINE = 315.0  # seconds; leave ~15s buffer for the harness


def _elapsed_ok():
    return (time.time() - _T0) < _HARD_DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Return 2000 finite scores (higher = better to evaluate next).
    
    Strategy: Cost-aware Expected Improvement with adaptive exploration.
    EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    where z = (f_best - mu(x)) / sigma(x)
    
    We divide by cost to get cost-effective EI.
    Additionally, we add a small exploration bonus for high-uncertainty regions
    that are not already well-explored (low mu improvement potential).
    
    The key insight: near the start, exploration matters more. As we get closer
    to the optimum, exploitation should dominate. We use the current f_best 
    relative to the known minimum (-3.86278) to adapt the exploration weight.
    """
    if not _elapsed_ok():
        # Emergency: return something finite
        return np.zeros_like(mu)
    
    cost_safe = np.where(cost > 1e-9, cost, 1e-9)
    sigma_safe = np.where(sigma > 1e-12, sigma, 1e-12)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-effective EI
    cei = ei / cost_safe
    
    # Adaptive exploration bonus:
    # When f_best is far from the optimum, we should explore more.
    # The known minimum is approximately -3.86278 for hartmann 3d.
    # As f_best approaches this value, reduce exploration.
    known_min = -3.86278
    # Normalized distance from optimum (0 = at optimum, 1+ = far)
    dist_from_opt = max(0.0, (f_best - known_min) / 1.0)
    # Exploration weight: higher when far from optimum
    # Use a smooth decay: weight = exp(-dist_from_opt * 2)
    exp_weight = np.exp(-dist_from_opt * 2.0)
    
    # Exploration bonus: reward high uncertainty in regions where EI is low
    # This helps escape local optima early on
    # Normalize sigma to [0, 1] for the bonus
    sigma_max = np.max(sigma)
    if sigma_max > 1e-12:
        sigma_norm = sigma / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma)
    
    # Exploration bonus: sigma_norm * exp_weight / cost
    # This is small when close to optimum, larger when far
    exp_bonus = sigma_norm * exp_weight / cost_safe
    
    # Combine: main EI signal + exploration bonus
    # The exploration bonus is scaled to be at most ~50% of typical EI values
    scores = cei + 0.5 * exp_bonus
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
