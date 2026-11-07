"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 325.0  # wall-clock seconds total allowed (safety margin before 330s kill)


def acquisition(mu, sigma, f_best):
    """
    LCB (Lower Confidence Bound) acquisition function for minimization.
    
    For minimization, we want to find points where the true function value
    is likely to be low. The LCB gives a lower bound on the function value:
        LCB = mu - kappa * sigma
    
    We maximize LCB (equivalently, minimize -LCB). Higher LCB means the
    lower bound is less negative / closer to zero, but for minimization we
    actually want to MINIMIZE the LCB (find the most negative lower bound).
    
    Wait - let me reconsider. For minimization:
    - We want to find x that minimizes f(x).
    - The posterior gives mu(x) ≈ E[f(x)] and sigma(x) = std(f(x)).
    - A lower confidence bound on f(x) is: LCB(x) = mu(x) - kappa * sigma(x)
    - If LCB(x) is very negative, it means f(x) could be very low (good for minimization).
    - So we want to MINIMIZE LCB(x), or equivalently MAXIMIZE -LCB(x).
    
    But the task says "higher meaning evaluate next", so we return -LCB or some
    transformation where higher = better for minimization.
    
    Actually, re-reading: "Return 2000 finite scores, higher meaning evaluate next"
    and "the candidate with the highest score" is evaluated.
    
    For minimization, we want to pick the point most likely to have a low f value.
    LCB = mu - kappa*sigma. A more negative LCB suggests potentially lower f.
    So we want to MINIMIZE LCB, which means we should return a score where lower LCB
    gets higher score. So score = -LCB = -mu + kappa*sigma.
    
    Wait, that's the same as UCB for minimization. Let me think again...
    
    Actually for MINIMIZATION:
    - UCB (Upper Confidence Bound) for minimization: we want to explore where f could be low.
    - The standard UCB for maximization is: UCB = mu + kappa*sigma (maximize this)
    - For minimization, we flip: we want to minimize, so we use LCB = mu - kappa*sigma
    - We want to find the point with the LOWEST LCB (most negative), meaning f could be very low.
    - So the acquisition score should be: score = -LCB = -(mu - kappa*sigma) = -mu + kappa*sigma
    - Higher score means more negative LCB means potentially lower f. Good.
    
    So score = -mu + kappa * sigma, and we pick the max.
    
    This is structurally different from the EI-based approach in the current program.
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        return -mu
    
    # Guard against negative or zero sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # LCB acquisition for minimization: score = -mu + kappa * sigma
    # Choose kappa adaptively based on the scale of the problem
    # Use a moderate exploration factor
    kappa = 2.0
    
    score = -mu + kappa * sigma_safe
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
