"""
Cost-aware acquisition function for Bayesian optimization (rastrigin 2d).
Uses Expected Improvement with a cost-aware exploration bonus.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import erf, exp

# Deadline guard constants
_START_TIME = time.time()
_DEADLINE_SECONDS = 285.0  # Safety margin before 300s limit

def _check_deadline():
    """Check if we are close to the time limit."""
    if time.time() - _START_TIME > _DEADLINE_SECONDS:
        return True
    return False

def _expected_improvement(mu, sigma, f_best):
    """
    Compute Expected Improvement (EI) for minimization.
    EI = E[max(f_best - f(x), 0)]
    
    For a Gaussian f(x) ~ N(mu, sigma^2):
    EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    where Phi is the standard normal CDF and phi is the standard normal PDF.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    diff = f_best - mu
    z = diff / sigma_safe
    
    # Standard normal CDF and PDF using scipy.special
    # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    # phi(z) = exp(-0.5 * z^2) / sqrt(2*pi)
    Phi = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    phi = exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    
    # Expected Improvement
    ei = diff * Phi + sigma_safe * phi
    
    # Ensure non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    return ei

def _ucb_score(mu, sigma, beta):
    """
    Upper Confidence Bound for minimization.
    UCB = mu - beta * sigma (minimize this, so maximize -mu + beta*sigma)
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    return -mu + beta * sigma_safe

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing rastrigin 2d.
    Uses Expected Improvement normalized by cost, with an exploration bonus.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Check deadline
    if _check_deadline():
        # Return a simple, fast score to avoid timeout
        cost_safe = np.maximum(cost, 1e-9)
        sigma_safe = np.maximum(sigma, 1e-9)
        scores = (-mu + 0.5 * sigma_safe) / cost_safe
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    # Compute Expected Improvement
    ei = _expected_improvement(mu, sigma, f_best)
    
    # Cost-aware normalization: divide by cost to prefer cheaper evaluations
    cost_safe = np.maximum(cost, 1e-9)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # EI normalized by cost
    ei_scores = ei / cost_safe
    
    # Add a small exploration bonus proportional to sigma/cost
    # This helps escape local optima in rastrigin's multimodal landscape
    exploration_bonus = 0.05 * sigma_safe / cost_safe
    
    # Combine EI and exploration bonus
    scores = ei_scores + exploration_bonus
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
