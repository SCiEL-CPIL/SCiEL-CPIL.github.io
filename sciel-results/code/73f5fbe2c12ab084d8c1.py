"""Rewritten program with a stronger cost-aware acquisition function."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Deadline guard
_DEADLINE = time.time() + 320.0  # safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    mu, sigma: shape (2000,) - GP mean and std (in objective units)
    f_best: scalar - lowest observed value
    cost: shape (2000,) - evaluation cost for each candidate
    
    Returns: shape (2000,) - scores (higher = evaluate next)
    """
    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Guard against zero/negative costs
    cost = np.maximum(cost, 1e-9)
    
    # Guard against zero/negative sigma
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute delta = f_best - mu (expected improvement baseline)
    delta = f_best - mu
    
    # Compute z = delta / sigma
    z = delta / sigma
    
    # Expected Improvement (EI)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    ei = delta * Phi_z + sigma * phi_z
    ei = np.maximum(ei, 0.0)
    
    # Expected Cost-Adjusted Improvement: EI / cost
    ei_cost = ei / cost
    
    # Uncertainty-based exploration term: sigma / cost
    # This encourages exploration in high-uncertainty, low-cost regions
    ucb_term = sigma / cost
    
    # Adaptive exploration weight based on the spread of mu values
    # Early on (high uncertainty), explore more; later, exploit more
    mu_std = np.std(mu)
    if mu_std > 1e-6:
        # Normalize: when mu_std is large relative to f_best scale, explore more
        # Use a sigmoid-like transition
        exploration_scale = np.abs(f_best) + 1.0  # scale reference
        w_explore = 0.3 + 0.4 * np.exp(-mu_std / (exploration_scale * 0.1))
    else:
        w_explore = 0.3
    
    # Also consider the ratio of best EI to mean EI to adapt
    ei_mean = np.mean(ei_cost)
    if ei_mean > 1e-9:
        # If EI is concentrated, exploit more
        ei_max = np.max(ei_cost)
        concentration = ei_max / (ei_mean + 1e-9)
        # High concentration -> lower exploration
        w_explore *= (1.0 / (1.0 + 0.3 * (concentration - 1.0)))
        w_explore = np.clip(w_explore, 0.1, 0.8)
    
    score = ei_cost + w_explore * ucb_term
    
    # Add a small bonus for very low-cost candidates to ensure cheap points are considered
    # This helps in early stages where cost varies significantly
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    if cost_max > cost_min + 1e-9:
        cost_norm = (cost - cost_min) / (cost_max - cost_min)
        # Small bonus for low cost
        score += 0.05 * (1.0 - cost_norm)
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
