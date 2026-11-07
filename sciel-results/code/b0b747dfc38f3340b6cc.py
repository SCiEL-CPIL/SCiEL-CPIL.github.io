import numpy as np
import time
from scipy.stats import norm

# Track time for safety margin
_START_TIME = time.time()
_DEADLINE = _START_TIME + 315.0  # safety margin before 330s wall-clock


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on Levy 2D.
    
    Strategy: Use a quantile-based feasibility gate on cost-efficiency (EI/cost),
    combined with a blend of EI and UCB for the objective score. This forces
    exploitation of low-cost regions early, then naturally relaxes as the GP
    uncertainty structure changes.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        return np.ones_like(mu, dtype=float) * 0.0

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    # Guard against non-finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    # Compute Expected Improvement (EI) for minimization
    improvement = f_best - mu
    z = improvement / sigma
    z = np.clip(z, -30.0, 30.0)
    EI = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    EI = np.where(np.isfinite(EI), EI, 0.0)
    EI = np.maximum(EI, 0.0)

    # Compute UCB for minimization: -mu + beta * sigma
    # Use adaptive beta based on how close we are to convergence
    # When f_best is close to 0 (minimum), we want more exploitation (lower beta)
    # When f_best is far from 0, we want more exploration (higher beta)
    beta = 2.0 + 2.0 * np.exp(-abs(f_best) / 0.1)
    UCB = -mu + beta * sigma
    UCB = np.where(np.isfinite(UCB), UCB, 0.0)

    # Blend EI and UCB: use EI when we have good improvement estimates,
    # UCB for exploration when uncertainty is high
    # Weight based on relative uncertainty
    uncertainty_ratio = sigma / (np.abs(mu) + 1e-8)
    ucb_weight = np.clip(uncertainty_ratio / (uncertainty_ratio + 1.0), 0.2, 0.8)
    blended = (1.0 - ucb_weight) * EI + ucb_weight * UCB
    blended = np.where(np.isfinite(blended), blended, 0.0)

    # Cost efficiency: blended score per unit cost
    eff = blended / cost
    eff = np.where(np.isfinite(eff), eff, 0.0)

    # Quantile-based feasibility gate
    # Use the 25th percentile as the threshold - candidates below this are dampened
    # This is more robust than mean-based gating
    q25 = np.percentile(eff, 25)
    
    if not np.isfinite(q25) or q25 <= 0:
        # Fallback: use plain blended score
        scores = blended
    else:
        # Gate: keep candidates at or above the 25th percentile of efficiency
        # Damp below-threshold candidates, but don't zero them out completely
        gate = q25
        feasible = eff >= gate
        # Damp factor for infeasible candidates
        damp = 0.15
        scores = np.where(feasible, blended, blended * damp)

    # Ensure finite and add small floor
    scores = np.where(np.isfinite(scores), scores, 0.0)
    floor = 1e-12
    scores = np.maximum(scores, floor)

    return scores
