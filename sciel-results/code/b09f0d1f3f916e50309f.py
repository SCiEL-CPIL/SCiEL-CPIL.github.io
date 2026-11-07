# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 285.0  # Safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition blending Expected Improvement with UCB.
    
    Key insights:
    1. Blend EI (exploitation) with UCB (exploration) for balanced search.
    2. Use sublinear cost penalty (1/sqrt(cost)) to allow exploration of 
       high-cost regions that may contain the global minimum.
    3. Adaptive beta for UCB based on f_best to balance exploration/exploitation.
    """
    if time.time() > _DEADLINE:
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma = np.maximum(sigma, 1e-9)
        cost = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma
        ei = sigma * (z * norm.cdf(z) + norm.pdf(z))
        ucb = -mu + 1.5 * sigma
        cost_penalty = 1.0 / np.sqrt(cost)
        scores = (ei + 0.5 * ucb) * cost_penalty
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)

    # Sublinear cost penalty: 1/sqrt(cost) allows exploration of high-cost regions
    cost_penalty = 1.0 / np.sqrt(cost)

    # Expected Improvement (minimization: improvement = f_best - f(x))
    z = (f_best - mu) / sigma
    ei = sigma * (z * norm.cdf(z) + norm.pdf(z))
    # EI is already in objective units, non-negative

    # UCB (Lower Confidence Bound for minimization)
    # Adaptive beta: larger when f_best is poor (early), smaller when f_best is good (late)
    # f_best for powell 4d min is 0, so f_best > 0 means we haven't found min yet
    # Scale beta based on how far f_best is from 0
    beta = 1.5 + 0.5 * np.clip(f_best / (f_best + 1.0), 0.0, 1.0)
    ucb = -mu + beta * sigma

    # Blend EI and UCB with cost-aware scaling
    # EI handles exploitation, UCB handles exploration
    # Both are in objective units (lower is better for ucb, higher is better for ei)
    # We want higher scores = more likely to evaluate next
    # EI: higher is better (more expected improvement)
    # UCB: lower is better (lower predicted value with uncertainty)
    # Convert UCB to a score: higher is better means we want low ucb values
    # So use -ucb as the score component, or just use ucb and negate
    
    # Actually, let's think about this more carefully:
    # For minimization, we want to find points with low f(x)
    # EI = E[max(f_best - f(x), 0)] -- higher EI means more expected improvement
    # UCB = mu - beta*sigma -- lower UCB means more likely to be good
    
    # So for the acquisition score (higher = evaluate next):
    # - EI directly: higher is better
    # - For UCB: we want low UCB, so use -ucb or (some_constant - ucb)
    
    # Let's use a normalized blend:
    # score = w1 * EI + w2 * (-UCB) * scale_factor
    # But EI and UCB have different scales, so let's normalize
    
    # Simpler approach: use EI as primary, add UCB as exploration bonus
    # The UCB term should encourage exploring regions with low mu + high sigma
    
    # Let's use: score = EI + alpha * sigma * (1 - mu/scale)
    # But this is getting complicated. Let's stick with the proven approach:
    
    # Blend: score = EI + weight * UCB_exploration
    # Where UCB_exploration encourages low mu and high sigma
    
    # Actually, the inspiration says blending EI with UCB (beta=1.5) works well.
    # Let me use: score = EI + 0.5 * (-mu + beta * sigma)
    # But -mu can be negative, so let's shift it
    
    # Better: score = EI + 0.5 * beta * sigma - 0.5 * mu
    # This combines exploitation (EI, -mu) with exploration (sigma)
    
    w_ei = 1.0
    w_ucb = 0.5
    
    # UCB score: -mu + beta * sigma (lower mu and higher sigma give higher score)
    ucb_score = -mu + beta * sigma
    
    scores = (w_ei * ei + w_ucb * ucb_score) * cost_penalty

    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
