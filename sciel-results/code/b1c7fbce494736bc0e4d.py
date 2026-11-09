import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill

def _check_deadline():
    """Check if we're approaching the deadline."""
    return (time.time() - _start_time) > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a structural change: Cost-Adjusted Probability of Improvement (PoI)
    combined with a GP uncertainty bonus, rather than simple EI/cost.
    
    The key insight: dividing EI by cost over-penalizes expensive regions.
    Instead, we use PoI/cost (probability of success per unit budget) plus
    a sigma-based exploration bonus that helps in the small-budget regime.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP std predictions
    f_best : float - best observed value so far
    cost : array of shape (2000,) - evaluation costs
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = better)
    """
    # Deadline guard
    if _check_deadline():
        # Return a simple fallback: just pick the cheapest with some uncertainty
        return sigma / cost
    
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Cost-aware: normalize costs to get a relative measure
    # cost ranges from 1.0 to 2.0 (1 + mean(x_scaled))
    # We want to prefer cheaper evaluations
    
    # Compute z-score for Probability of Improvement
    z = (f_best - mu) / sigma_safe
    
    # Probability of Improvement: P(f(x) < f_best)
    # For minimization, improvement means f(x) < f_best
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    poi = norm.cdf(z)
    
    # Clip to avoid numerical issues
    poi = np.clip(poi, 1e-10, 1.0 - 1e-10)
    
    # Cost-Adjusted PoI: probability of improvement per unit cost
    # This explicitly prioritizes candidates with high likelihood of success
    # per unit of budget spent
    cost_adjusted_poi = poi / cost
    
    # Exploration bonus: GP uncertainty scaled appropriately
    # In the small-budget regime (~25 evaluations), exploration is important
    # Use sigma normalized by the scale of the problem
    # The three-hump camel has values roughly in [0, 3], so scale ~1
    # Use a moderate exploration coefficient
    exploration_bonus = 0.15 * sigma_safe / cost
    
    # Combine: cost-adjusted PoI dominates, exploration helps escape local optima
    scores = cost_adjusted_poi + exploration_bonus
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
