"""Cost-aware acquisition function for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Hartmann 3D.
    
    Combines Expected Improvement with cost-awareness and adaptive exploration.
    Uses EI/cost^alpha for cost-awareness and a UCB-like exploration term.
    """
    # Ensure inputs are arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: ensure finite values
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Standard EI computation
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Clip EI to non-negative
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware adjustment:
    # Use EI / cost^alpha where alpha controls cost sensitivity
    # cost is in [1, 2] since it's 1 + mean(x) where x in [0,1]^3
    # alpha=0.5 gives moderate cost penalty
    alpha = 0.5
    cost_factor = 1.0 / np.power(cost, alpha)
    
    # Adaptive exploration: use UCB-like term
    # beta adapts based on how close we are to the optimum
    # When EI is small (close to optimum), increase exploration
    # When EI is large (far from optimum), decrease exploration
    ei_max = np.max(ei)
    if ei_max > 1e-9:
        exploration_ratio = np.clip(1.0 - ei / ei_max, 0.0, 1.0)
    else:
        exploration_ratio = np.ones_like(ei)
    
    # Beta: higher when exploration_ratio is high (EI is small)
    # Scale beta to be meaningful relative to sigma
    beta = 0.5 + 1.5 * exploration_ratio
    
    # UCB term: mu + beta * sigma (for minimization, we want high sigma regions)
    ucb_bonus = beta * sigma
    
    # Combine: cost-aware EI plus exploration bonus
    score = ei * cost_factor + ucb_bonus * 0.1
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
