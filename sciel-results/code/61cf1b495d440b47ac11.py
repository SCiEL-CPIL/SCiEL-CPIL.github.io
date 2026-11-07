"""Cost-aware acquisition for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_DEADLINE = _START_TIME + 295  # safety margin before 300s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: combine Expected Improvement (EI) with a
    cost penalty derived from the expected marginal gain per unit cost.
    
    Args:
        mu: (2000,) GP mean predictions
        sigma: (2000,) GP std predictions
        f_best: float, best observed (lowest) value
        cost: (2000,) evaluation cost per candidate
    
    Returns:
        scores: (2000,) finite scores, higher = pick next
    """
    if time.time() > _DEADLINE:
        # Fallback: prefer cheap points with some uncertainty
        return 1.0 / (cost + 1e-9)
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardize: z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi = norm.cdf, phi = norm.pdf
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Clip negative EI to zero (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: we want high EI per unit cost.
    # Use a softened denominator to avoid extreme weighting of very cheap points
    # while still strongly preferring low-cost evaluations.
    # cost ranges from 1 to 2 (since 1 + mean(x_scaled) where x_scaled in [0,1])
    # A reasonable scaling: divide EI by cost, but use a power < 1 to soften.
    # Try: score = EI / cost^0.5  (soft cost penalty)
    # Or: score = EI * exp(-cost)  (exponential decay)
    # Let's use a hybrid: EI / (cost * (1 + sigma_safe * 0.1))
    # This penalizes both high cost and high uncertainty (redundant exploration)
    
    # Actually, for Rastrigin with a known global minimum at 0, and f_best likely
    # still well above 0 early on, we want to explore regions with high EI.
    # The cost structure: cost = 1 + mean(x/10.24 + 0.5) so cost in [1, 2].
    # Points near the center of the domain are more expensive.
    
    # Use: score = EI / cost  (simplest cost-aware)
    scores = ei / (cost + 1e-9)
    
    # Add a small exploration bonus for high uncertainty to avoid premature convergence
    # to a local minimum (Rastrigin has many local minima)
    # bonus = sigma_safe * 0.05  (small)
    # But this might hurt exploitation. Let's keep it pure EI/cost first.
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
