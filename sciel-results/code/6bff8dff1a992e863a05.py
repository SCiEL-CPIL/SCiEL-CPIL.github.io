import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 285.0  # Safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Powell 4D.
    
    Strategy:
    - Primary exploitation: Expected Improvement (EI) for minimization
    - Exploration: UCB term with adaptive beta that increases as f_best improves
    - Cost-awareness: sublinear penalty 1/sqrt(cost) to allow exploration of 
      higher-cost regions where the global minimum may reside
    - The blend balances finding new minima (EI) with exploring uncertain regions (UCB)
    """
    # Fast path if we're running out of time
    if time.time() > _DEADLINE:
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma = np.maximum(sigma, 1e-9)
        cost = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma
        ei = sigma * (z * norm.cdf(z) + norm.pdf(z))
        scores = (ei + 0.5 * (-mu + 1.5 * sigma)) / np.sqrt(cost)
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Numerical stability
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)

    # Sublinear cost penalty: 1/sqrt(cost) allows exploration of high-cost regions
    # while still preferring cheaper evaluations
    cost_penalty = 1.0 / np.sqrt(cost)

    # Expected Improvement for minimization
    # EI = E[max(f_best - f(x), 0)] = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI is non-negative by definition

    # Adaptive UCB for minimization
    # UCB score = -mu + beta * sigma (higher is better)
    # Beta adapts: larger when f_best is poor (early, need exploration),
    # smaller when f_best is good (late, focus on exploitation)
    # f_best for Powell 4D min is 0, so f_best > 0 means we haven't found min yet
    # Use a smooth transition: beta = 1.0 + 1.0 * sigmoid-like function of f_best
    # When f_best is large (early), beta ~ 2.0 (more exploration)
    # When f_best is small (late), beta ~ 1.0 (less exploration)
    # Simple approximation: beta = 1.0 + 1.0 / (1.0 + f_best)
    # When f_best=0: beta=2.0, when f_best=10: beta~1.09
    beta = 1.0 + 1.0 / (1.0 + np.clip(f_best, 0.0, None))
    ucb_score = -mu + beta * sigma

    # Blend EI (exploitation) and UCB (exploration)
    # EI naturally balances probability and magnitude of improvement
    # UCB adds direct uncertainty-driven exploration
    # Weight: EI gets 1.0, UCB gets 0.5 to keep EI as primary signal
    score = ei + 0.5 * ucb_score

    # Apply cost-aware penalty
    scores = score * cost_penalty

    # Ensure outputs are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
