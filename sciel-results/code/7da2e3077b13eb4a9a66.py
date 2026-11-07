"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Cost-Weighted UCB.
    
    Switches from Expected Improvement to Upper Confidence Bound (UCB).
    This structural change treats the problem as an optimistic search,
    where the score is the potential improvement (f_best - mu) plus an
    exploration bonus proportional to uncertainty (sigma), all normalized
    by the evaluation cost. This naturally penalizes 'crowded' (low sigma)
    regions and 'expensive' regions, promoting diverse and cost-effective
    exploration.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions.
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed objective value (lower is better).
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-10)
    
    # UCB Parameter
    # kappa controls the exploration-exploitation trade-off.
    # A value of 1.0 is a standard starting point for normalized objectives.
    kappa = 1.0
    
    # Exploitation term: Potential improvement
    # Positive if mu < f_best
    exploitation = f_best - mu
    
    # Exploration term: Uncertainty bonus
    # High sigma indicates unexplored (sparse) regions
    exploration = kappa * sigma_safe
    
    # Combined UCB value
    ucb_value = exploitation + exploration
    
    # Cost-aware: divide by cost
    score = ucb_value / cost_safe
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
