"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UCB acquisition function for Bayesian optimization.
    
    Uses Upper Confidence Bound with a fixed exploration constant,
    divided by cost to balance exploration, exploitation, and evaluation cost.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions at candidate points
    sigma : array-like, shape (2000,)
        GP standard deviation predictions at candidate points
    f_best : float
        Best (lowest) observed function value
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division issues
    sigma = np.maximum(sigma, 1e-9)
    
    # UCB: exploit + explore
    # Since we're minimizing, we want low mu, so UCB for minimization is:
    # UCB_min(x) = mu(x) - beta * sigma(x)
    # But for acquisition (higher is better to select), we negate or use:
    # score = -(mu - beta * sigma) / cost = (beta * sigma - mu) / cost
    
    beta = 2.0  # Exploration constant
    
    # For minimization: we prefer low mu and high uncertainty
    # UCB for minimization: ucb = mu - beta * sigma (lower is better for selection)
    # But we want higher acquisition score = better, so:
    # score = -(mu - beta * sigma) / cost = (beta * sigma - mu) / cost
    
    # However, we need to be careful: if mu is very low (good), we want high score.
    # Let's think again:
    # In minimization, a point is good if mu is low.
    # UCB for minimization typically uses: lower bound = mu - beta * sigma
    # We want to select points with low lower bound (optimistic about finding low values).
    # So acquisition score should be high when mu - beta * sigma is low.
    # score = -(mu - beta * sigma) / cost = (beta * sigma - mu) / cost
    
    # But this can be problematic if mu is large negative. Let's use a different approach:
    # Use the standard UCB form adapted for cost-aware minimization.
    
    # Actually, let's reconsider. The standard approach for minimization:
    # We want to find x that minimizes f(x).
    # GP predicts f(x) ~ N(mu(x), sigma(x)^2)
    # UCB for minimization: we're optimistic, so we think f(x) could be as low as mu(x) - beta * sigma(x)
    # We want to select x with the lowest such optimistic value.
    # So acquisition score (higher = better) = -(mu(x) - beta * sigma(x)) / cost(x)
    # = (beta * sigma(x) - mu(x)) / cost(x)
    
    # This is what we have. Let's compute it.
    
    ucb_term = beta * sigma - mu
    scores = ucb_term / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
