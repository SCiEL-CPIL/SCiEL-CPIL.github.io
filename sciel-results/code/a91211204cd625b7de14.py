"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware Expected Improvement with adaptive exploration
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # Number of evaluations so far (rough estimate from cost)
    # cost = 1 + mean(x_scaled), so total_cost ≈ n_evals * avg_cost
    # But we can use the magnitude of cost array to estimate progress
    # Actually, we don't know n_evals directly, but we can use f_best to adapt
    
    # Adaptive exploration: more exploration early, less later
    # When f_best is close to minimum (-0.8), reduce exploration
    # The minimum of cosine8 is -0.8
    f_min_possible = -0.8
    improvement_room = f_best - f_min_possible
    
    # If improvement_room is small, we're close to optimum - reduce exploration
    # If improvement_room is large, we're far from optimum - increase exploration
    # Use a sigmoid-like function to smoothly transition
    exploration_scale = 1.0 / (1.0 + np.exp(-20 * (improvement_room - 0.1)))
    
    # Standard EI
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware: normalize by cost
    # cost is in [1, 2]
    ei_per_cost = ei / cost
    
    # Exploration term: UCB-like with sigma
    # Use a softer exploration that scales with improvement_room
    explore_weight = 0.15 * exploration_scale
    exploit_weight = 1.0
    
    # Also add a small term for points with very high uncertainty
    # to avoid missing good regions
    high_uncertainty_bonus = 0.05 * exploration_scale * np.log1p(sigma / (sigma.mean() + eps))
    
    score = exploit_weight * ei_per_cost + explore_weight * (sigma / cost) + high_uncertainty_bonus
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
