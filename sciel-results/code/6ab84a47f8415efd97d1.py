"""Cost-aware acquisition for Ackley 2D BO with adaptive UCB-like hybrid scoring and deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0

def _check_deadline():
    return time.time() > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    if _check_deadline():
        mu_safe = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
        cost_safe = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e6)
        with np.errstate(divide='ignore', invalid='ignore'):
            fallback_scores = -mu_safe / (cost_safe + 1e-9)
        return np.nan_to_num(fallback_scores, nan=0.0, posinf=0.0, neginf=0.0)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=0.0)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1e6)

    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-6)

    # Adaptive UCB: tune beta based on exploration progress
    # Early: more exploration (higher beta), Late: more exploitation (lower beta)
    # Use the fraction of cost consumed as a proxy for progress
    total_cost = float(np.sum(cost))
    
    # Normalize: early stages have low total_cost, late stages have high
    # Budget is 30, so total_cost / 30 gives fraction of budget used
    progress = min(max(total_cost / 30.0, 0.0), 1.0)
    
    # Beta decays from 1.5 (early) to 0.5 (late)
    beta = 1.5 - 1.0 * progress
    
    # UCB-like hybrid: exploit (f_best - mu) and explore (beta * sigma)
    raw_scores = (f_best - mu) + beta * sigma
    
    # Cost-aware: divide by cost
    scores = raw_scores / cost

    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    return scores
# EVOLVE-BLOCK-END
