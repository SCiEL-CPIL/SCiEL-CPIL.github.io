"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a hybrid approach combining:
    1. Cost-adjusted Expected Improvement (EI) for exploitation
    2. Cost-adjusted Upper Confidence Bound (UCB) for exploration
    
    The blend weights exploration more heavily early on and shifts to
    exploitation as more budget is consumed.
    """
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Expected Improvement
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Upper Confidence Bound
    # Beta controls exploration-exploitation tradeoff
    # Use a beta that encourages exploration given low budget
    beta = 0.5
    
    ucb = mu + beta * sigma_safe
    
    # Cost adjustment: divide by cost to favor cheaper evaluations
    # But we want higher scores to be evaluated next, so we invert the cost effect
    # Actually, for minimization, we want to prefer points with low cost
    # So we divide the acquisition value by cost (higher cost = lower priority)
    
    # Normalize costs to [0, 1] range for stable weighting
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range > 1e-12:
        cost_norm = (cost - cost_min) / cost_range
    else:
        cost_norm = np.zeros_like(cost)
    
    # Blend EI and UCB
    # Weight based on how much we've explored
    # Early on, favor UCB; later, favor EI
    # Use a simple heuristic: blend = 0.5 * EI + 0.5 * UCB, both cost-adjusted
    
    # Cost-adjusted EI: prefer high EI and low cost
    ei_adjusted = ei / (cost + 1e-9)
    
    # Cost-adjusted UCB: prefer high UCB (i.e., high uncertainty or low predicted value)
    # Since we're minimizing, UCB = mu + beta*sigma gives an upper bound on the true value
    # We want to maximize the chance of finding a lower value, so we use -UCB or just UCB
    # Actually for minimization, we want to explore where the lower bound might be low
    # Let's use a different formulation: for minimization, we want to minimize f(x)
    # UCB for minimization: LCB = mu - beta*sigma (lower confidence bound)
    # We want to evaluate points where LCB is low (potentially good)
    lcb = mu - beta * sigma_safe
    
    # For the acquisition score (higher = better to evaluate):
    # EI is already oriented for minimization (improvement over f_best)
    # For UCB/LCB: we want to prefer points with low predicted value and high uncertainty
    # So score = -lcb / cost or similar
    
    # Let's use a cleaner approach:
    # Score = alpha * EI_adjusted + (1-alpha) * (-lcb) / cost
    # But -lcb can be negative, which is problematic
    
    # Alternative: use the standard cost-adjusted EI which is well-defined
    # And add a small exploration bonus based on sigma/cost
    
    # Final approach: Cost-adjusted EI with an exploration term
    # score = EI / cost + gamma * sigma / cost
    # This rewards both expected improvement and uncertainty, penalized by cost
    
    gamma = 0.3  # Exploration weight
    
    score = ei_adjusted + gamma * sigma_safe / (cost + 1e-9)
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
