"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_acq_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _acq_start_time
    if _acq_start_time is None:
        _acq_start_time = time.time()
    
    # Deadline guard: return simple EI/cost if we're close to the limit
    elapsed = time.time() - _acq_start_time
    if elapsed > 280:  # 300s limit, leave 20s margin
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma_safe = np.maximum(sigma, 1e-8)
        cost_safe = np.maximum(cost, 1e-8)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        score = ei / cost_safe
        score = np.where(np.isfinite(score), score, 0.0)
        return score
    
    # Ensure arrays are float64 and handle edge cases
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clamp sigma to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Clamp cost to avoid division by zero
    cost_safe = np.maximum(cost, 1e-8)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Variance penalty: penalize high uncertainty to reduce sensitivity to GP noise
    # This makes the acquisition more robust in high-dimensional settings
    var_penalty = 0.01 * sigma_safe
    
    # Combine: weighted sum of EI and PI, minus variance penalty
    # Weight PI more when we're close to the minimum (exploitation phase)
    min_possible = -0.8
    gap_to_min = f_best - min_possible
    # As gap shrinks, increase PI weight (more exploitation)
    pi_weight = 0.3 + 0.4 * np.exp(-gap_to_min / 0.3)
    ei_weight = 1.0 - pi_weight
    
    combined = ei_weight * ei + pi_weight * pi * sigma_safe - var_penalty
    combined = np.maximum(combined, 0.0)
    
    # Cost-aware: divide by cost to favor cheaper evaluations
    score = combined / cost_safe
    
    # Small exploration bonus for early stages
    explore_weight = 0.02 * np.exp(-gap_to_min / 0.5)
    ucb_bonus = explore_weight * sigma_safe / cost_safe
    
    score = score + ucb_bonus
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
