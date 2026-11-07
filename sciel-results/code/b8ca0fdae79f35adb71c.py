"""Rewritten program: rank-based cost-aware acquisition for Rastrigin 2D BO."""
import numpy as np
import time
from scipy.stats import norm

_START_TIME = time.time()
_DEADLINE_SECONDS = 320.0  # Safety margin before 330s wall-clock limit

def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE_SECONDS:
        return np.zeros_like(mu, dtype=np.float64)

    # Sanitize inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=1e6, neginf=-1e6)
    cost = np.nan_to_num(cost, nan=1.5, posinf=2.0, neginf=1.0)

    # Cost is in [1, 2] (1 + mean of x scaled to [0,1])
    cost = np.clip(cost, 1.0, 2.0)
    sigma = np.maximum(sigma, 1e-12)

    # Expected Improvement for minimization
    z = (f_best - mu) / sigma
    z = np.clip(z, -50, 50)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # UCB-like exploration bonus (beta controls exploration intensity)
    # Use a moderate beta to encourage exploring uncertain regions
    beta = 2.0
    ucb = mu + beta * sigma  # for minimization, we want low mu + high sigma

    # Convert to "improvement potential" score (higher = better)
    # EI already captures improvement; add exploration via sigma-weighted term
    # exploration_bonus = sigma * (1 - norm.cdf(z))  # tail probability
    # But simpler: use EI + gamma * sigma to blend exploitation and exploration
    gamma = 0.5
    combined = ei + gamma * sigma

    # Cost-aware gating:
    # Instead of dividing by cost^alpha (which can amplify noise in cheap regions),
    # use a soft gate that heavily penalizes expensive candidates.
    # cost_gate = exp(-lambda * (cost - 1)) so cost=1 -> 1, cost=2 -> exp(-lambda)
    # With lambda=3: cost=1 -> 1.0, cost=1.5 -> 0.22, cost=2 -> 0.05
    # This strongly suppresses high-cost regions while preserving cheap ones
    lam = 3.0
    cost_gate = np.exp(-lam * (cost - 1.0))

    score = combined * cost_gate

    # Ensure finite output
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)

    return score
