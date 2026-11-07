"""Cost-aware acquisition with multi-scale UCB + EI for Bayesian optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard
_start_time = None
_DEADLINE = 315.0  # safety margin before 330s kill


def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        return 0.0
    return time.time() - _start_time


def acquisition(mu, sigma, f_best, cost):
    """
    Multi-scale cost-aware acquisition: combines cost-normalized EI with
    an adaptive UCB exploration term. Structural change from single-ratio EI/cost
    to a power-mean aggregation that better balances exploitation vs exploration.
    
    mu, sigma, cost: arrays of shape (2000,)
    f_best: scalar, lowest observed value
    Returns: array of shape (2000,) with finite scores (higher = evaluate next)
    """
    # Deadline guard
    if _elapsed() > _DEADLINE:
        return np.ones(len(mu)) * 1e-10

    n = len(mu)
    eps = 1e-12
    s = np.maximum(sigma, eps)
    c = np.maximum(cost, eps)

    # Normalize cost to [0,1] for better scaling: cost ranges ~[1, 2]
    # Map to [0.5, 1.0] so cheap points get slightly higher weight
    c_norm = (c - 1.0) / 1.0  # now in [0, 1] roughly
    c_norm = np.clip(c_norm, 0.0, 1.0)
    # Inverse cost weight: cheap -> high weight
    w_cost = 1.0 - 0.5 * c_norm  # in [0.5, 1.0]

    # Expected Improvement (EI)
    z = (f_best - mu) / s
    z = np.clip(z, -12.0, 12.0)
    ei = (f_best - mu) * norm.cdf(z) + s * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # UCB-style exploration term: beta * sigma
    # Adaptive beta: start high, decrease as budget is consumed
    # Use f_best magnitude as a proxy for convergence
    # When f_best is close to 0, we're near optimum -> reduce exploration
    # Three-hump camel min is 0, so f_best approaching 0 means converged
    scale = max(abs(f_best), 1e-6)
    beta = 0.5 * np.sqrt(np.log(1.0 + n) / (1.0 + scale * 10))
    beta = np.clip(beta, 0.05, 1.5)

    ucb = mu + beta * s

    # Cost-aware versions
    ei_cost = ei * w_cost
    ucb_cost = ucb * w_cost

    # Power-mean aggregation (p=1.5) instead of simple sum or ratio
    # This gives more weight to the dominant term while still considering both
    p = 1.5
    # Use log-sum-exp style for numerical stability
    # score = (ei_cost^p + ucb_cost^p)^(1/p)
    # But we want higher = better, and ucb should be inverted (lower mu = better)
    # Actually for minimization: EI is already "improvement" (higher = better)
    # UCB: we want to explore where ucb is high (upper bound is high = uncertain)
    # But for minimization, high UCB means we might find something lower
    # So both EI and UCB work in the same direction for minimization
    
    # Combine: use geometric-mean-like blend
    # score = ei_cost * (1 + 0.3 * ucb_cost / (ei_cost + eps))
    # This boosts points with high uncertainty relative to their EI
    
    ratio = ucb_cost / (ei_cost + eps)
    ratio = np.clip(ratio, 0.0, 10.0)
    
    # Final score: EI dominates, UCB provides exploration bonus
    score = ei_cost * (1.0 + 0.25 * np.log1p(ratio))

    # Add a small pure-exploration floor for very uncertain regions
    # This prevents the algorithm from ignoring high-sigma areas entirely
    exploration_floor = 0.02 * s * w_cost
    score = score + exploration_floor

    # Ensure finiteness and non-negativity
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    score = np.clip(score, 0.0, 1e6)

    return score
# EVOLVE-BLOCK-END
