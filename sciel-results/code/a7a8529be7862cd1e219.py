"""Cost-aware acquisition for Hartmann-6d BO loop."""
import numpy as np
from scipy.stats import norm
import time

_DEADLINE = 315.0  # seconds, safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    t_start = time.time()

    # Guard against zero/negative sigma
    sigma_safe = np.maximum(sigma, 1e-10)

    # Cost-aware normalization: divide by cost^beta to prefer cheaper points
    beta = 1.3
    cost_safe = np.maximum(cost, 1e-6)
    cost_factor = cost_safe ** beta

    # Estimate remaining budget and scale the improvement threshold
    # total_cost = mean(cost) as a proxy for current spend
    total_cost = np.mean(cost)
    remaining_budget = max(30.0 - total_cost, 1.0)
    
    # Number of evaluations so far is roughly total_cost (since each eval costs ~1+mean(x))
    # We estimate n_evals as total_cost / 1.5 (average cost per eval in [1,2])
    n_evals_so_far = max(total_cost / 1.5, 1.0)
    n_evals_remaining = max(20.0 - n_evals_so_far, 1.0)
    
    # Dynamic gamma: scale the threshold by remaining budget
    # As budget depletes, we demand larger improvements
    gamma = 0.1 * (remaining_budget / 20.0)
    
    # Target threshold: f_best - gamma * (remaining_budget / n_evals_remaining)
    # This forces the search to demand larger relative improvements as budget depletes
    threshold = f_best - gamma * (remaining_budget / n_evals_remaining)
    
    # Standardize z-score relative to the dynamic threshold
    z = (threshold - mu) / sigma_safe

    # Probability of Improvement (POI) with dynamic threshold
    poi = norm.cdf(z)
    
    # Expected Improvement (EI) with dynamic threshold
    ei = (threshold - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Combine EI (exploitation) and POI (exploration)
    w_ei = 0.7
    w_poi = 1.0 - w_ei
    base_score = (w_ei * ei + w_poi * poi) / cost_factor
    
    # Add a UCB-like exploration term scaled by uncertainty
    ucb_weight = 0.1
    ucb_term = ucb_weight * sigma_safe / cost_factor
    scores = base_score + ucb_term

    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Deadline check
    if time.time() - t_start > _DEADLINE:
        return np.zeros_like(mu)

    return scores
