"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Track global start time for deadline safety
_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # seconds to spare before 330s kill

# Constants
MIN_POSSIBLE = -0.8
MAX_POSSIBLE = 1.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Expected Improvement (EI) with adaptive exploration.
    
    Structural Change:
    Replaces Adaptive UCB with Cost-Adjusted Expected Improvement.
    EI is a more principled Bayesian acquisition function that explicitly models the
    expected improvement over the current best value, weighted by the probability of
    improvement. This tends to be more robust in high-dimensional spaces than UCB,
    which can be overly exploratory or exploitative depending on the beta parameter.
    
    The cost adjustment divides the improvement by the evaluation cost, making the
    acquisition function aware of the expense of each evaluation.
    
    Additionally, we add a small exploration bonus based on sigma to ensure that
    regions with high uncertainty are still explored, but in a more controlled way
    than UCB.
    """
    # Deadline guard
    if time.time() - _START_TIME > 330.0 - _DEADLINE_MARGIN:
        return np.zeros_like(mu)

    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-8)
    cost_safe = np.maximum(cost, 1.0)
    
    # Calculate z-score: how many sigma away f_best is from the predicted mean
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement for minimization:
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is the CDF and phi is the PDF of the standard normal distribution.
    # This is the expected improvement over f_best if we evaluate at x.
    
    # Compute Phi(z) and phi(z)
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    
    # EI for minimization
    EI = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (it should be, but numerical issues can cause small negatives)
    EI = np.maximum(EI, 0.0)
    
    # Cost-adjusted EI
    score = EI / cost_safe
    
    # Add a small exploration bonus to encourage exploration in uncertain regions
    # The bonus is proportional to sigma but scaled down to not dominate the EI term
    exploration_bonus = 0.1 * sigma_safe / cost_safe
    
    score = score + exploration_bonus
    
    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
