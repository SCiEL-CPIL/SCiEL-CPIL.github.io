"""Cost-aware LCB acquisition with adaptive exploration for Bayesian optimization on Ackley 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function with adaptive exploration.
    
    The exploration parameter kappa is adapted based on f_best:
    - Early (high f_best): higher kappa -> more exploration
    - Late (low f_best): lower kappa -> more exploitation
    
    The score is the optimistic improvement (f_best - (mu - kappa*sigma)) divided by cost.
    """
    # Ensure sigma is positive
    s = np.maximum(sigma, 1e-9)
    
    # Adaptive exploration parameter kappa
    # f_best is the minimum observed value (lower is better)
    # When f_best is high (early), we want to explore more -> high kappa
    # When f_best is low (late), we want to exploit more -> low kappa
    alpha = 0.5
    kappa_min = 1.0
    kappa_max = 2.5
    # progress goes from 0 (f_best=0, late) to 1 (f_best=inf, early)
    progress = 1.0 - np.exp(-alpha * f_best)
    progress = np.clip(progress, 0.0, 1.0)
    kappa = kappa_min + (kappa_max - kappa_min) * progress
    
    # LCB: optimistic estimate of the minimum
    lcb = mu - kappa * s
    
    # Potential improvement: how much better than f_best we expect
    improvement = f_best - lcb
    
    # Cost awareness: divide by cost
    c = np.maximum(cost, 1e-9)
    scores = improvement / c
    
    # Safety: Ensure finite and non-negative
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
