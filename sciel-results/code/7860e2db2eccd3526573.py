import numpy as np
from scipy.stats import norm
import time

_START_TIME = None
_DEADLINE_SAFETY = 5.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using dynamic cost-normalized EI with adaptive UCB.
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed value so far
    cost : array of shape (2000,) - evaluation costs
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition values (higher = better)
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    elapsed = time.time() - _START_TIME
    # If we're close to the deadline, return a simple score quickly
    if elapsed > 320.0:
        # Emergency: just return EI/cost
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        cost_safe = np.maximum(cost, 1e-9)
        score = ei / cost_safe
        score = np.where(np.isfinite(score), score, 0.0)
        return score
    
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Dynamic cost normalization: use marginal cost relative to the minimum cost
    # This favors points that offer high improvement per unit of *additional* cost
    cost_min = np.min(cost_safe)
    # Marginal cost: how much more expensive this point is compared to the cheapest
    marginal_cost = cost_safe - cost_min
    # Add a small floor to avoid division by zero
    marginal_cost_safe = np.maximum(marginal_cost, 0.1 * cost_min)
    
    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Cost-normalized EI: divide by marginal cost
    base_score = ei / marginal_cost_safe
    
    # Adaptive exploration term: UCB with time-adaptive beta
    # beta increases exploration early, decreases later
    # Use a simple schedule: beta starts high and decays
    # Since we don't know the exact iteration, use elapsed time as a proxy
    # Early: high exploration, Late: low exploration
    total_time_budget = 300.0  # approximate
    time_fraction = min(elapsed / total_time_budget, 1.0)
    # Beta: starts at 2.0, decays to 0.5
    beta = 0.5 + 1.5 * (1.0 - time_fraction)
    
    # UCB term: mu + beta * sigma (for minimization, we want low mu, so use -mu + beta*sigma)
    # Actually for minimization, UCB is: mu - beta * sigma (we want to minimize)
    # But acquisition should be higher = better, so: score = -mu + beta * sigma
    # However, we already have EI which handles exploitation. Let's add an exploration bonus.
    # Exploration bonus: beta * sigma / marginal_cost_safe
    exploration_bonus = beta * sigma_safe / marginal_cost_safe
    
    # Combine
    score = base_score + exploration_bonus
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
