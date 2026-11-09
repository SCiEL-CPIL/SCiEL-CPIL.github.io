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
    Acquisition function using Upper Confidence Bound (UCB).
    
    This is a structural change from the previous Kriging-Believer/EI hybrid.
    UCB directly balances exploitation (low mu) and exploration (high sigma)
    via a linear combination: score = -mu + beta * sigma.
    
    The parameter beta is adapted over the optimization steps:
    - Early steps: high beta to encourage exploration of uncertain regions.
    - Late steps: low beta to focus on exploitation of known good regions.
    
    Args:
        mu: array of shape (2000,) - posterior mean
        sigma: array of shape (2000,) - posterior standard deviation
        f_best: float - lowest observed value
    
    Returns:
        array of shape (2000,) - scores, higher means evaluate next
    """
    global _eval_count
    
    # Deadline guard: fall back to simple exploitation (negative mean)
    if time.time() - _T0 > _DEADLINE:
        return -mu
    
    # Increment evaluation counter
    _eval_count += 1
    
    # Total BO steps is 25 (30 evaluations - 5 initial)
    total_steps = 25
    progress = min(_eval_count / float(total_steps), 1.0)
    
    # Adaptive beta for UCB
    # Start with high exploration (beta=2.0), end with low exploration (beta=0.5)
    # Linear decay
    beta_start = 2.0
    beta_end = 0.5
    beta = beta_start + (beta_end - beta_start) * progress
    
    # UCB score for minimization: maximize (-mu + beta * sigma)
    score = -mu + beta * sigma
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
