"""Rewritten cost-aware acquisition: budget-aware, cost-biased UCB."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    # mu, sigma, cost: shape (2000,). Return 2000 finite scores (higher = evaluate next).
    sigma = np.maximum(sigma, 1e-9)

    # Budget-aware exploration: as remaining budget shrinks, trust the model less.
    remaining = np.clip(30.0 - float(np.mean(cost)), 0.0, 30.0)
    beta = 1.2 + 2.5 * (remaining / 30.0)

    # UCB score (minimization): high mu + beta * sigma.
    score = mu + beta * sigma

    # Cost bias: prefer cheap candidates (lower mean(x)); cost = 1 + mean(x) in [0,1].
    # This is the only place cost changes the *ranking* (not just magnitude),
    # because cheap points let the loop spend the fixed budget on more evaluations.
    score = score - 0.35 * cost

    # Safety: keep everything finite.
    score = np.nan_to_num(score, nan=0.0, posinf=1e12, neginf=-1e12)
    return score
# EVOLVE-BLOCK-END
