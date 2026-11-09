"""Rewritten program with a stronger cost-aware acquisition for rastrigin 2d."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def _expected_improvement(mu, sigma, f_best):
    """Standard EI with a small floor on sigma to avoid division by zero."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return ei


def _ucb(mu, sigma, f_best, beta):
    """Upper confidence bound (maximization form)."""
    return mu + beta * sigma


def _pi(mu, sigma, f_best):
    """Probability of improvement."""
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    return norm.cdf(z)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing rastrigin 2d.
    
    Parameters:
    -----------
    mu : array_like, shape (2000,)
        GP mean predictions.
    sigma : array_like, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far.
    cost : array_like, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x_scaled)).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher means evaluate next.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-9)
    
    # Compute multiple acquisition components
    ei = _expected_improvement(mu, sigma, f_best)
    pi = _pi(mu, sigma, f_best)
    
    # Adaptive beta for UCB: balance exploration and exploitation
    # Use a moderate beta that scales with uncertainty
    beta = 1.5
    ucb = _ucb(mu, sigma, f_best, beta)
    
    # Combine EI and UCB with cost normalization
    # Normalize each component by its max to make them comparable
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    ucb_max = np.max(ucb) if np.max(ucb) > 0 else 1.0
    
    ei_norm = ei / ei_max
    ucb_norm = ucb / ucb_max
    
    # Blend: 60% cost-normalized EI, 40% cost-normalized UCB
    # This balances exploitation (EI) with exploration (UCB)
    combined = 0.6 * ei_norm + 0.4 * ucb_norm
    
    # Cost-aware normalization: divide by cost to prefer cheaper evaluations
    scores = combined / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
