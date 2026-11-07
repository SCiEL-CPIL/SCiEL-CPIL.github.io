"""Cost-aware acquisition function for Bayesian optimization (levy 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Track total elapsed time for deadline guard
_T0 = time.time()
_DEADLINE = 315.0  # 300s limit + 15s safety margin


def _time_left():
    return _DEADLINE - (time.time() - _T0)


def acquisition(mu, sigma, f_best, cost):
    """
    Pareto-optimal cost-normalized acquisition combining EI and UCB.
    
    Uses marginal cost normalization: ratio of acquisition value to the
    marginal cost above the minimum observed cost, explicitly favoring
    low-cost points with high expected improvement per unit additional expense.
    
    Parameters
    ----------
    mu : np.ndarray, shape (N,)
        GP mean predictions.
    sigma : np.ndarray, shape (N,)
        GP standard deviation predictions.
    f_best : float
        Best observed objective value (lower is better).
    cost : np.ndarray, shape (N,)
        Cost of evaluating each candidate.
    
    Returns
    -------
    np.ndarray, shape (N,)
        Acquisition scores (higher = evaluate next).
    """
    # Deadline guard: if we've been running too long, return a simple, fast score
    if _time_left() < 5.0:
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-9)
        cost_safe = np.maximum(cost, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        return ei / cost_safe

    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma and cost to small positive values
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate z-score
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -10.0, 10.0)
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    ei = np.maximum(ei, 0.0)
    
    # UCB component for exploration
    # For minimization: -mu + beta * sigma rewards low mean and high uncertainty
    beta = 1.5
    ucb = -mu + beta * sigma_safe
    
    # Combine EI and UCB with weighted sum
    # EI is in objective units, UCB is also in objective units
    # Weight UCB lower to prioritize exploitation
    w_ucb = 0.25
    combined = ei + w_ucb * np.maximum(ucb, 0.0)
    
    # Pareto-optimal marginal cost normalization
    # Instead of dividing by absolute cost, divide by marginal cost above min cost
    # This explicitly favors low-cost points that offer high improvement per unit
    # of additional expense
    cost_min = np.min(cost_safe)
    cost_max = np.max(cost_safe)
    
    if cost_max - cost_min > 1e-12:
        # Marginal cost: how much more expensive this point is vs the cheapest
        marginal_cost = cost_safe - cost_min
        # Add a small floor to avoid division by zero for the cheapest points
        marginal_cost_safe = np.maximum(marginal_cost, 0.05 * (cost_max - cost_min))
        # Normalize to [0, 1] range for consistency
        marginal_cost_norm = marginal_cost_safe / (cost_max - cost_min)
    else:
        # All costs are the same, no normalization needed
        marginal_cost_norm = np.ones_like(cost_safe)
    
    # Cost-aware score: acquisition value per unit marginal cost
    # Higher score = more improvement per unit of additional expense
    score = combined / (marginal_cost_norm + 0.1)
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    score = np.maximum(score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
