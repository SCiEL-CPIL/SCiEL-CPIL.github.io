"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = None
_DEADLINE_LIMIT = 310.0  # Safety margin before 330s kill

def _init_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + _DEADLINE_LIMIT

def _check_deadline():
    global _DEADLINE
    if _DEADLINE is not None and time.time() > _DEADLINE:
        return True
    return False

def acquisition(mu, sigma, f_best, cost):
    _init_deadline()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive and cost is positive
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1e-6)
    
    # Structural change: Cost-Adjusted Lower Confidence Bound (LCB)
    # LCB = mu - kappa * sigma
    # kappa is dynamically increased as total cost is consumed,
    # shifting from aggressive exploration (high kappa) to precise exploitation (low kappa).
    
    # Estimate budget consumption: cost values are 1 + mean(x_scaled_to_[0,1])
    # So cost ranges from 1.0 to 2.0. Total budget is 30.
    # Use the mean cost of evaluated points to estimate how much budget has been used.
    # Since we don't have explicit "number of evaluations", we infer from cost distribution.
    # A simpler approach: use the median cost as a proxy for average cost per eval,
    # and estimate remaining budget fraction.
    
    # Better approach: track cumulative cost implicitly.
    # The cost array contains costs of the 2000 candidate points being scored.
    # We need to estimate how much of the 30-unit budget has been consumed.
    # 
    # Key insight: f_best is the lowest observed value. As we converge, f_best approaches 0.
    # We can use the "progress" toward the known minimum (0 for Griewank) as a signal,
    # but that's not directly budget consumption.
    #
    # Alternative: Use the range of mu values and the fact that early on, mu is spread out,
    # while later mu concentrates near the minimum.
    #
    # Simplest robust approach: Use a fixed schedule based on the spread of sigma.
    # Early: high sigma everywhere -> high kappa
    # Late: low sigma near minimum, high sigma elsewhere -> lower kappa
    #
    # Actually, the cleanest approach: use the quantile of sigma to determine phase.
    # If median sigma is high, we're early -> explore more.
    # If median sigma is low, we're late -> exploit more.
    
    med_sigma = np.median(sigma_safe)
    max_sigma = np.max(sigma_safe)
    
    # Normalize: 0 = fully explored (low sigma), 1 = fully unexplored (high sigma)
    if max_sigma > 1e-12:
        exploration_phase = med_sigma / max_sigma
    else:
        exploration_phase = 0.0
    
    # Clip to [0, 1]
    exploration_phase = np.clip(exploration_phase, 0.0, 1.0)
    
    # Dynamic kappa: 
    # Early (high exploration_phase): kappa ~ 3.0 (aggressive exploration)
    # Late (low exploration_phase): kappa ~ 0.5 (precise exploitation)
    # Use a smooth interpolation
    kappa_min = 0.5
    kappa_max = 3.0
    kappa = kappa_min + (kappa_max - kappa_min) * exploration_phase
    
    # LCB score: mu - kappa * sigma (we want to minimize this, so we negate for "higher is better")
    lcb = mu - kappa * sigma_safe
    
    # Cost-aware: divide by cost to prefer cheaper evaluations
    # Higher score = better, so we want (negative LCB) / cost
    # Since lower LCB is better, we use -lcb / cost
    scores = -lcb / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    if _check_deadline():
        return scores
    
    return scores
# EVOLVE-BLOCK-END
