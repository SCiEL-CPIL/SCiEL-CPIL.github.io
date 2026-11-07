import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition function for Bayesian optimization."""
    t_start = time.time()
    deadline = 325.0 - 15.0  # 15 second safety margin

    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Safety: clamp sigma and cost to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)

    # Expected Improvement
    improvement = f_best - mu
    z = improvement / sigma_safe
    # Handle z overflow/underflow
    z = np.clip(z, -50, 50)

    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Cost-aware scoring: EI / cost with adaptive exploration
    # The key insight: use a stronger exploration term that scales with
    # the current state of the optimization (how close we are to convergence)
    
    # Measure how "converged" we are by looking at the distribution of improvements
    # If EI is small everywhere, we need more exploration
    ei_max = np.max(ei) if len(ei) > 0 else 0.0
    ei_mean = np.mean(ei) if len(ei) > 0 else 0.0
    
    # Adaptive exploration weight: increase exploration when EI is small
    # This helps escape local optima in multi-modal functions
    # The weight is between 0.02 and 0.15 based on convergence state
    if ei_max > 1e-8:
        convergence_ratio = ei_mean / (ei_max + 1e-10)
        # When convergence_ratio is low, we're converged and need more exploration
        exploration_weight = 0.02 + 0.13 * (1.0 - np.clip(convergence_ratio, 0, 1))
    else:
        exploration_weight = 0.15
    
    # Cost-aware exploration bonus
    # Use sqrt(sigma) to balance exploration vs exploitation better
    exploration_bonus = exploration_weight * np.sqrt(sigma_safe) / np.sqrt(cost_safe)
    
    # Main score: EI / cost
    score = ei / cost_safe
    
    # Combine with exploration
    total_score = score + exploration_bonus
    
    # Add a small UCB-like term for additional diversity
    # This helps in multi-modal landscapes
    ucb_weight = 0.03
    ucb_term = ucb_weight * sigma_safe / cost_safe
    total_score = total_score + ucb_term

    # Ensure finite output
    total_score = np.where(np.isfinite(total_score), total_score, 0.0)
    
    # Check if we're approaching the deadline
    elapsed = time.time() - t_start
    if elapsed > deadline:
        # Return a safe default to avoid timeout
        return np.zeros_like(total_score)

    return total_score
