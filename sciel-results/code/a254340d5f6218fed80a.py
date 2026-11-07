"""Cost-aware acquisition for BO with cost(x) = 1 + mean(x_scaled_to_[0,1])."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import erfc
from scipy.stats import norm
import time as _time

_DEADLINE_S = 285.0
_START = None


def _now():
    return _time.time()


def _elapsed():
    if _START is None:
        return 0.0
    return _now() - _START


def _ei_fast(mu, sigma, f_best):
    """Fast Expected Improvement for minimization using erfc for stability."""
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    d = f_best - mu
    z = d / sigma_safe
    z = np.clip(z, -50.0, 50.0)
    
    # norm.cdf(z) = 0.5 * erfc(-z / sqrt(2))
    cdf = 0.5 * erfc(-z / np.sqrt(2.0))
    pdf = norm.pdf(z)
    
    ei = d * cdf + sigma_safe * pdf
    
    # Where sigma is very small, EI is essentially max(d, 0)
    ei = np.where(sigma < eps, np.maximum(d, 0.0), ei)
    return ei


def acquisition(mu, sigma, f_best, cost):
    global _START
    if _START is None:
        _START = _now()

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    # Safety: guarantee finite inputs.
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e12, neginf=-1e12)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e3, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)

    n = mu.shape[0]

    # Deadline guard: if we are too close to the kill, return a cheap
    # exploitation-only score so we never blow the budget mid-call.
    if _elapsed() > _DEADLINE_S:
        eps = 1e-9
        exploit = np.maximum(f_best - mu, 0.0)
        score = exploit / (cost + eps)
        return np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)

    # Compute Expected Improvement
    ei = _ei_fast(mu, sigma, f_best)
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=0.0)

    # Cost-aware scaling
    # cost is in [1, 2]. 
    # We want to penalize high cost. 
    # A common approach is EI / cost^alpha.
    # alpha=1 is linear penalty. alpha<1 is less punitive.
    # Let's try a slightly softer penalty to allow exploring expensive areas if EI is high,
    # but still strongly prefer cheap points when EI is similar.
    # However, the cost function is 1 + mean(x_scaled). 
    # In the center of the box, cost is ~1.5. At corners, cost is ~1 or ~2.
    # Actually, x_scaled is in [0,1]. mean is in [0,1]. cost is in [1,2].
    
    cost_safe = np.maximum(cost, 1.0)
    
    # Structural change: Use a "regret-aware" cost discount.
    # Instead of just dividing by cost, we can use a hyper-parameter 
    # that adapts to the current best value.
    # But simpler: just tune the power.
    # Previous program used 0.6. 
    # Let's try a slightly stronger penalty to be more cost-efficient, 
    # as the budget is tight (30 cost units).
    # If we spend too much on expensive points, we run out of budget.
    # Average cost is ~1.5. 30/1.5 = 20 evaluations.
    # We start with 5 random. So ~15 left.
    
    # Let's try alpha = 0.8.
    alpha_cost = 0.8
    cost_factor = cost_safe ** alpha_cost
    
    ei_cost = ei / cost_factor

    # Exploration boost
    # The previous program used 0.05 * sigma / cost.
    # This can be problematic if sigma is large in unexplored regions,
    # leading to random exploration.
    # A better exploration term is one that balances with exploitation.
    # UCB: mu - kappa * sigma (for minimization, we want low mu, high sigma is good for uncertainty)
    # But we are maximizing acquisition.
    # Acquisition = EI / cost.
    # EI already includes an exploration component (sigma * pdf).
    # Adding another sigma term might over-explore.
    
    # Let's try a smaller exploration boost, or none, and rely on EI.
    # Or use a "probability of improvement" (PI) component?
    # PI = norm.cdf(z).
    # PI is bounded [0,1]. EI can be large.
    
    # Let's stick to EI but maybe add a small noise to break ties?
    # No, deterministic is better for reproducibility within a run.
    
    # Let's try a slightly different exploration term:
    # boost = beta * sigma / (1 + sigma) to saturate.
    # Or just rely on EI.
    
    # Let's try reducing the exploration boost significantly.
    explore_boost = 0.01 * sigma / cost_safe
    
    score = ei_cost + explore_boost
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    return score
# EVOLVE-BLOCK-END
