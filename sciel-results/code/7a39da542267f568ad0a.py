# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses Expected Improvement (EI) normalized by cost, with a small
    exploration bonus to avoid premature convergence.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions
    sigma : array-like, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x))
    
    Returns:
    --------
    scores : array, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Deadline guard: if we're taking too long, return a safe default
    # This is a static function so we don't track time here, but the
    # computation is vectorized and fast.
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute z-score for Expected Improvement
    improvement = f_best - mu
    z = improvement / sigma_safe
    
    # EI formula: E[max(0, f_best - f(x))]
    # = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure non-negative (numerical safety)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware: divide by cost
    # Add a small epsilon to cost to avoid division by zero
    cost_safe = np.maximum(cost, 1e-10)
    
    # EI per unit cost
    score = ei / cost_safe
    
    # Add a small exploration bonus based on sigma to encourage
    # visiting uncertain regions, but keep it small so exploitation
    # dominates when EI is significant
    # The bonus is also cost-normalized
    exploration_weight = 0.05
    exploration_bonus = exploration_weight * sigma_safe / cost_safe
    
    # Combine: primarily EI/cost, with small exploration term
    total_score = score + exploration_bonus
    
    # Ensure all values are finite
    total_score = np.where(np.isfinite(total_score), total_score, 0.0)
    
    return total_score
# EVOLVE-BLOCK-END
