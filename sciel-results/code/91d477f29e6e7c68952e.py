"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    Primary: Expected Improvement per unit cost.
    Refinement: Small UCB component for exploration in high-uncertainty, low-cost regions.
    """
    start_time = time.time()
    
    # Ensure arrays are float64 for precision
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: clip sigma to avoid division by zero or negative values
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Calculate Expected Improvement (EI)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # EI should be non-negative, but due to numerical errors it might be slightly negative
    ei = np.maximum(ei, 0.0)
    
    # Calculate UCB (Upper Confidence Bound) for exploration
    # We use a small beta to not dominate the EI term
    beta = 0.5
    ucb = -mu + beta * sigma_safe  # Maximizing -mu + beta*sigma is equivalent to minimizing mu - beta*sigma
    
    # Cost-aware weighting
    # The primary driver is EI / cost
    score_ei = ei / cost
    
    # The exploration term is scaled by cost to prefer cheaper explorations
    # We use a small weight to ensure it doesn't override the exploitation signal
    # Normalizing cost to a 0-1 range helps with stability
    cost_min = np.min(cost)
    cost_max = np.max(cost)
    cost_range = cost_max - cost_min
    if cost_range > 1e-12:
        cost_norm = (cost - cost_min) / cost_range
    else:
        cost_norm = np.zeros_like(cost)
    
    # Weight the UCB term: higher weight for cheaper points
    # We use an exponential decay to smooth the preference
    cost_weight = np.exp(-cost_norm)
    
    # Scale UCB to be comparable to EI/cost
    # EI/cost has units of [value]/[cost]. UCB has units of [value].
    # We need to scale UCB by 1/mean(cost) or similar to make them comparable.
    # Let's use the inverse of the mean cost as a scale factor.
    mean_cost = np.mean(cost)
    ucb_scaled = ucb / mean_cost
    
    # Final score: 95% EI/cost, 5% UCB/cost_weighted
    # The 5% is small enough to not disrupt the strong EI signal but enough to help escape local minima
    score = 0.95 * score_ei + 0.05 * ucb_scaled * cost_weight
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Deadline guard: if we've spent too much time, return the result
    # This is a single pass, so it should be very fast, but just in case
    elapsed = time.time() - start_time
    if elapsed > 320:
        # Fallback to pure EI/cost if something went wrong
        return ei / cost
        
    return score
# EVOLVE-BLOCK-END
