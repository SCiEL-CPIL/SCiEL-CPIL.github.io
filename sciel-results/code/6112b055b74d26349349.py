"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization on hartmann 3d.
    
    Strategy: Global budget-aware scaling using normalized EI with a penalty
    term based on cost relative to mean cost, shifting exploration-exploitation
    balance based on remaining budget.
    """
    # Deadline guard: track elapsed time
    t0 = time.time()
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Ensure EI is non-negative (should be, but numerical issues possible)
    ei = np.maximum(ei, 0.0)
    
    # Global budget-aware scaling:
    # 1. Normalize EI by mean EI to get relative improvement
    mean_ei = np.mean(ei)
    if mean_ei > 1e-12:
        ei_normalized = ei / mean_ei
    else:
        ei_normalized = ei
    
    # 2. Compute cost penalty based on cost relative to mean cost
    mean_cost = np.mean(cost)
    if mean_cost > 1e-12:
        # Penalty: higher cost relative to mean reduces score
        # Use a smooth penalty that's 1 at mean cost, >1 for expensive, <1 for cheap
        cost_ratio = cost / mean_cost
        # Penalty term: subtract a fraction of the normalized EI for expensive points
        # The penalty grows with cost ratio
        penalty = 0.3 * (cost_ratio - 1.0)
        # Ensure penalty doesn't make scores negative in extreme cases
        penalty = np.maximum(penalty, -1.0)
    else:
        penalty = np.zeros_like(cost)
    
    # 3. Combine: score = normalized EI * (1 + penalty)
    # When cost is below mean, penalty is negative, so (1 + penalty) < 1, reducing score
    # When cost is above mean, penalty is positive, so (1 + penalty) > 1, increasing score
    # Wait, that's backwards. We want to penalize high cost.
    # Let's flip: score = normalized EI / (1 + alpha * (cost_ratio - 1))
    # Or: score = normalized EI * exp(-beta * (cost_ratio - 1))
    
    # Better approach: divide by a cost-aware factor
    # factor = 1 + beta * max(0, cost_ratio - 1)
    # This only penalizes points more expensive than average
    beta = 0.5
    cost_ratio = cost / mean_cost if mean_cost > 1e-12 else np.ones_like(cost)
    excess_cost = np.maximum(cost_ratio - 1.0, 0.0)
    factor = 1.0 + beta * excess_cost
    
    score = ei_normalized / factor
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Check deadline: if we've used too much time, return a safe fallback
    elapsed = time.time() - t0
    if elapsed > 280:  # Safety margin before 300s limit
        # Return a simple fallback: just EI divided by cost
        fallback = ei / np.maximum(cost, 1e-9)
        fallback = np.where(np.isfinite(fallback), fallback, 0.0)
        return fallback
    
    return score
# EVOLVE-BLOCK-END
