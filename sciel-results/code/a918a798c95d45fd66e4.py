"""Bayesian optimization acquisition for 2D RBF mixture with Matérn-5/2 GP."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = 320.0
_EPS = 1e-12

# Track evaluation count
_eval_count = 0

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Adaptive Upper Confidence Bound (UCB).
    
    Structural change: Replaces the Kriging-Believer + EI blend with a pure
    UCB strategy. UCB is defined as score = -mu + beta * sigma.
    We use an adaptive beta that starts high (exploration) and decays
    (exploitation) as the number of evaluations increases.
    
    Args:
        mu: array of shape (2000,) - posterior mean
        sigma: array of shape (2000,) - posterior standard deviation
        f_best: float - lowest observed value
    
    Returns:
        array of shape (2000,) - scores, higher means evaluate next
    """
    global _eval_count
    
    # Deadline guard: fall back to simple exploitation (negated mean)
    if time.time() - _T0 > _DEADLINE:
        return -mu
    
    # Increment evaluation counter
    _eval_count += 1
    
    sigma_safe = np.maximum(sigma, _EPS)
    
    # Total BO steps after initial points: 30 - 5 = 25
    total_steps = 25
    if _eval_count > total_steps:
        progress = 1.0
    else:
        progress = _eval_count / float(total_steps)
        
    # Adaptive Beta for UCB
    # Start with high exploration (beta=2.0) and decay to low exploration (beta=0.5)
    # Using a smooth decay curve
    beta_start = 2.0
    beta_end = 0.5
    # Cosine annealing for smooth transition
    beta = beta_end + 0.5 * (beta_start - beta_end) * (1 + np.cos(np.pi * progress))
    
    # UCB Score: We want to minimize mu - beta*sigma
    # So we maximize -mu + beta*sigma
    score = -mu + beta * sigma_safe
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
