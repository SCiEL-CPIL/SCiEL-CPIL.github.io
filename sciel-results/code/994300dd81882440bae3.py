"""Cost-aware acquisition for Bayesian optimization (hartmann 3d).
Uses a probability-of-improvement (PI) acquisition with cost normalization
and an adaptive exploration bonus. The PI component directly estimates the
probability that a candidate beats f_best, while the cost normalization
penalizes expensive evaluations. An adaptive sigma-weighted term provides
exploration that decays as the budget is consumed.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def _pi_score(mu, sigma, f_best):
    """Probability of improvement: P(f(x) < f_best)."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    return norm.cdf(z)


def _cost_normalized_ei(mu, sigma, f_best, cost, beta=1.0):
    """Expected improvement divided by cost^beta."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    cost_safe = np.maximum(cost, 1e-9)
    return ei / (cost_safe ** beta)


def _adaptive_exploration(mu, sigma, cost, f_best, budget_fraction):
    """Exploration bonus that decays with budget consumption.
    
    Uses a sigma-weighted term scaled by remaining budget fraction,
    encouraging exploration early and exploitation later.
    """
    # budget_fraction is in [0, 1], 1 = full budget remaining, 0 = exhausted
    # Exploration weight decays quadratically to favor exploitation late
    explore_weight = 0.5 * budget_fraction ** 1.5
    
    # sigma term: high uncertainty = high exploration value
    # Normalize sigma by typical scale (use median as robust estimate)
    sigma_med = np.median(sigma)
    if sigma_med < 1e-9:
        sigma_med = 1.0
    
    # Exploration score: high sigma relative to typical, discounted by cost
    cost_safe = np.maximum(cost, 1e-9)
    cost_med = np.median(cost_safe)
    if cost_med < 1e-9:
        cost_med = 1.0
    
    # Relative uncertainty (how much more uncertain than typical)
    rel_uncertainty = sigma / sigma_med
    
    # Cost penalty: more expensive = less attractive for exploration
    cost_penalty = 1.0 / (cost_safe / cost_med)
    
    # Combined exploration score
    exploration = explore_weight * rel_uncertainty * cost_penalty
    return exploration


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition function using PI + cost-normalized EI + adaptive exploration.
    
    Parameters:
        mu: array of shape (2000,) - GP mean
        sigma: array of shape (2000,) - GP std dev
        f_best: scalar - best observed value
        cost: array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
        array of shape (2000,) - finite scores, higher = evaluate next
    """
    # Deadline guard
    t_start = time.time()
    deadline = t_start + 290.0
    
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)
    
    n = len(mu)
    if n == 0:
        return np.array([], dtype=np.float64)
    
    # Safety: handle non-finite values
    sigma_safe = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    mu_safe = np.where(np.isfinite(mu), mu, f_best + 1.0)
    cost_safe = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Estimate budget fraction from cost: total budget is 30, initial cost ~5*1.5=7.5
    # Current total cost is approximately sum of evaluated costs, but we only have
    # per-candidate costs. Use median cost as proxy for typical evaluation cost.
    # The number of evaluations so far can be inferred from how many points have
    # been evaluated (not directly available). Use a heuristic: assume we're
    # roughly halfway through the budget on average.
    # Better: use the range of costs to estimate. Low-cost points are cheap,
    # high-cost points are expensive. The budget fraction remaining depends on
    # how much we've spent, which we don't track here. Use a fixed moderate value
    # that balances exploration and exploitation.
    # Alternative: use the minimum cost as a proxy for "early" and max for "late"
    cost_min = np.min(cost_safe)
    cost_max = np.max(cost_safe)
    cost_range = cost_max - cost_min
    if cost_range < 1e-9:
        budget_fraction = 0.5
    else:
        # Points with lower cost are "cheaper" to explore; use median position
        cost_med = np.median(cost_safe)
        budget_fraction = 1.0 - (cost_med - cost_min) / cost_range
        budget_fraction = np.clip(budget_fraction, 0.1, 0.9)
    
    # Compute component scores
    pi = _pi_score(mu_safe, sigma_safe, f_best)
    ei_cost = _cost_normalized_ei(mu_safe, sigma_safe, f_best, cost_safe, beta=1.0)
    exploration = _adaptive_exploration(mu_safe, sigma_safe, cost_safe, f_best, budget_fraction)
    
    # Normalize each component to [0, 1] range for fair combination
    # PI is already in [0, 1]
    pi_norm = pi
    
    # EI: normalize by max (if max > 0)
    ei_max = np.max(ei_cost)
    if ei_max > 1e-9:
        ei_norm = ei_cost / ei_max
    else:
        ei_norm = np.zeros_like(ei_cost)
    
    # Exploration: normalize by max
    exp_max = np.max(exploration)
    if exp_max > 1e-9:
        exp_norm = exploration / exp_max
    else:
        exp_norm = np.zeros_like(exploration)
    
    # Combine: PI is the primary signal (probability of beating best),
    # EI adds magnitude information, exploration encourages diversity
    # Weights: PI=0.5, EI=0.3, Exploration=0.2
    scores = 0.5 * pi_norm + 0.3 * ei_norm + 0.2 * exp_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
