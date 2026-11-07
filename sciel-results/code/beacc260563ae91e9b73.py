"""Rewritten acquisition: cost-normalized UCB with aggressive low-cost bias and safety margin."""
import time
import numpy as np
from scipy.stats import norm

# Track start time for deadline guard
_START_TIME = time.time()
_DEADLINE_MARGIN = 5.0  # seconds before 330s to stop heavy computation
_DEADLINE = _START_TIME + 330.0 - _DEADLINE_MARGIN


def _check_deadline():
    """Return True if we should stop heavy computation."""
    return (time.time() - _START_TIME) > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function: UCB normalized by cost^gamma.
    
    Parameters:
        mu: array of shape (2000,), GP mean predictions
        sigma: array of shape (2000,), GP std predictions
        f_best: scalar, best observed value so far
        cost: array of shape (2000,), cost of evaluating each candidate
    
    Returns:
        scores: array of shape (2000,), higher = evaluate next
    """
    # Safety: ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clamp sigma to avoid division by zero and negative values
    sigma = np.maximum(sigma, 1e-9)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-9)
    
    # UCB: balance exploitation (mu) and exploration (sigma)
    # Use a moderate kappa that adapts slightly to the scale of f_best
    # For minimization, UCB = mu - kappa * sigma (lower is better, so we want low mu + high sigma)
    # But the acquisition should be HIGH = good, so we negate: score = -(mu - kappa*sigma) = -mu + kappa*sigma
    # Alternatively, think of it as: we want to minimize mu, and uncertainty sigma suggests potential for improvement
    # Standard UCB for minimization: ucb = mu - kappa * sigma (lower ucb = more promising)
    # To make higher = better: score = -ucb = -mu + kappa * sigma
    
    # Choose kappa based on the problem scale. Since f_best is in objective units,
    # a reasonable kappa is on the order of the expected improvement scale.
    # Use a fixed kappa that works well for the test functions.
    kappa = 1.5
    
    # Compute UCB score (higher = more promising)
    ucb_score = -mu + kappa * sigma
    
    # Cost normalization: divide by cost^gamma to favor cheap evaluations
    # High gamma (e.g., 3.0) drastically favors low-cost regions
    gamma = 3.0
    cost_factor = np.power(cost, gamma)
    
    # Final score: UCB normalized by cost
    scores = ucb_score / cost_factor
    
    # Handle any non-finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
