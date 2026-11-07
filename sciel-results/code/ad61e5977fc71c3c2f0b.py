"""Cost-aware acquisition for BO with cost(x) = 1 + mean(x_scaled_to_[0,1])."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time as _time

_DEADLINE_S = 245.0
_START = None


def _now():
    return _time.time()


def _elapsed():
    if _START is None:
        return 0.0
    return _now() - _START


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

    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)

    # --- Expected Improvement (Gaussian) ---
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10.0, 10.0)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.where(sigma > 0, ei, np.maximum(f_best - mu, 0.0))
    ei = np.nan_to_num(ei, nan=0.0, posinf=1e6, neginf=0.0)

    # --- Upper Confidence Bound ---
    # Adaptive beta: start exploratory, become exploitative as budget depletes
    # cost is in [1, 2], starts at ~1.5 for uniform random, total budget 30
    # Use cost as a proxy for progress: early on cost is low (few evaluations)
    # Actually, we don't know total cost spent, but we can use the magnitude of f_best
    # or just use a fixed schedule based on the number of evaluations implied by cost
    
    # Better: use the spread of sigma to adapt. High sigma -> explore more.
    # Also, as f_best approaches the minimum, we should exploit more.
    # But we don't know the minimum.
    
    # Let's use a simple adaptive beta based on the relative uncertainty.
    # If sigma is large relative to the improvement, explore.
    # If sigma is small, exploit.
    
    # A common heuristic: beta = 1.0 + 0.5 * (1 - min_cost / max_cost)
    # But we don't have min/max cost directly.
    
    # Let's try a different approach: 
    # UCB = mu - beta * sigma
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    # Blend EI and UCB.
    # When sigma is high, UCB is good. When sigma is low, EI is good.
    
    # Let's use a fixed beta that works well in practice.
    beta = 1.5  # Moderate exploration
    
    ucb = mu - beta * sigma_safe
    ucb = np.nan_to_num(ucb, nan=1e12, posinf=1e12, neginf=-1e12)
    
    # Normalize UCB to be comparable to EI
    # UCB is in objective units (lower is better), EI is expected improvement (higher is better)
    # To make UCB a "score" (higher is better), we can use:
    # ucb_score = f_best - ucb = f_best - mu + beta * sigma
    # This is similar to EI but without the probability weighting.
    
    ucb_score = f_best - mu + beta * sigma_safe
    ucb_score = np.nan_to_num(ucb_score, nan=0.0, posinf=1e6, neginf=0.0)
    
    # --- Cost-aware scaling ---
    # cost is in [1, 2].
    # Use a power law for tunable cost sensitivity.
    cost_safe = np.maximum(cost, 1.0)
    
    # Adaptive cost sensitivity: 
    # Early on (low total cost), we can afford more expensive evaluations.
    # Later on (high total cost), we should prefer cheaper evaluations.
    # Since we don't track total cost, we use a heuristic:
    # If f_best is still high (far from optimum), explore more (lower cost sensitivity).
    # If f_best is low (close to optimum), exploit (higher cost sensitivity).
    
    # But we don't know the optimum. Let's use a fixed cost sensitivity.
    alpha_cost = 0.7  # Moderate cost sensitivity
    
    cost_factor = cost_safe ** alpha_cost
    
    ei_aware = ei / cost_factor
    ucb_aware = ucb_score / cost_factor
    
    # --- Adaptive blend ---
    # When sigma is high, trust UCB more. When sigma is low, trust EI more.
    # Use the relative sigma to determine the blend weight.
    
    # Normalize sigma to [0, 1] using a reference scale.
    # The length scale is 0.2 on [0,1]^d, so typical sigma might be ~0.1-0.5.
    sigma_ref = 0.3
    sigma_norm = sigma_safe / sigma_ref
    sigma_norm = np.clip(sigma_norm, 0.0, 2.0)
    
    # Weight for UCB increases with sigma.
    # When sigma_norm = 0, weight_ucb = 0.2 (mostly EI)
    # When sigma_norm = 2, weight_ucb = 0.8 (mostly UCB)
    weight_ucb = 0.2 + 0.3 * sigma_norm
    weight_ucb = np.clip(weight_ucb, 0.0, 1.0)
    weight_ei = 1.0 - weight_ucb
    
    # Final score
    score = weight_ei * ei_aware + weight_ucb * ucb_aware
    
    # Ensure finiteness.
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    return score
# EVOLVE-BLOCK-END
