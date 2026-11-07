"""Structural change: Thompson Sampling with cost-aware ranking and budget-scaled exploration."""
import time
import numpy as np

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin before 330s kill

# State for adaptive exploration
_call_count = 0
_last_f_best = None
_stagnation_counter = 0
_total_cost_seen = 0.0


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition: Thompson Sampling with cost-adjusted ranking
    and budget-scaled exploration.
    
    Uses a single random draw from the GP posterior, blends with the mean
    based on stagnation, and ranks by a cost-normalized score that includes
    a budget-decaying uncertainty bonus.
    
    Args:
        mu: array shape (2000,) - GP mean predictions in objective units
        sigma: array shape (2000,) - GP std dev predictions in objective units
        f_best: float - lowest observed value
        cost: array shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array shape (2000,) - acquisition scores (higher = evaluate next)
    """
    global _call_count, _last_f_best, _stagnation_counter, _total_cost_seen
    
    # Time guard: if we're running out of time, return a simple fallback
    if time.time() > _DEADLINE:
        c = np.maximum(cost, 1e-9)
        return -mu / c
    
    _call_count += 1
    
    # Track stagnation: if f_best hasn't improved, increase exploitation
    if _last_f_best is not None:
        if f_best < _last_f_best - 1e-8:
            _stagnation_counter = 0
        else:
            _stagnation_counter += 1
    _last_f_best = f_best
    
    # Track total cost to scale exploration with remaining budget
    _total_cost_seen = np.mean(cost) * _call_count  # approximate
    
    eps = 1e-9
    s = np.maximum(sigma, eps)
    c = np.maximum(cost, eps)
    
    # Thompson Sampling: draw a sample from the posterior
    rng = np.random.default_rng()
    sample = mu + s * rng.standard_normal(len(mu))
    
    # Adaptive kappa: reduce exploration when stagnant
    base_kappa = 0.4
    kappa = max(base_kappa * (1.0 / (1.0 + 0.15 * _stagnation_counter)), 0.1)
    
    # Blend: use a convex combination of sample and mean
    blended = kappa * sample + (1.0 - kappa) * mu
    
    # Cost-aware scoring: we want low function values and low cost
    score = -blended / c
    
    # Budget-scaled uncertainty bonus
    # Early on (low total cost), give more weight to uncertainty
    # As budget is consumed, reduce the bonus
    budget_fraction = min(_total_cost_seen / 30.0, 1.0)
    bonus_scale = 0.15 * (1.0 - 0.5 * budget_fraction)
    
    # Uncertainty bonus scaled by 1/cost so cheap uncertain points get a boost
    uncertainty_bonus = bonus_scale * s / c
    
    score = score + uncertainty_bonus
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
