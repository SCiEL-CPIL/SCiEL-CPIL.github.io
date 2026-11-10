"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320  # 10s safety margin before 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Effective Improvement (CEI) acquisition function.
    
    Maximizes expected improvement per unit of remaining budget, treating
    the optimization as a dynamic resource allocation problem.
    
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Deadline guard: if we're running out of time, use a fast fallback
    if time.time() > _DEADLINE:
        return -mu / np.maximum(cost, 1e-9)

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # 1. Compute Expected Improvement (EI) for minimization
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # 2. Estimate remaining budget dynamically
    # Total budget is 30 (from task: "While total cost is below 30")
    # The cost array contains the cost of each candidate point
    # We estimate cumulative cost from the number of evaluations already made
    # and the average cost observed so far
    
    # The cost array has shape (2000,) and contains costs for candidate points
    # We need to estimate what fraction of the budget has been used.
    # Since we don't have direct access to cumulative cost, we use heuristics:
    # - The mean cost of candidates gives us a sense of typical evaluation cost
    # - The number of evaluations is implicitly tracked by the GP state
    
    # Heuristic: estimate remaining budget based on the relationship between
    # f_best and the range of mu values (proxy for how much we've learned)
    
    # A simpler approach: use the mean cost as a proxy for typical cost,
    # and assume we're using a fraction of the budget proportional to
    # how concentrated our evaluations are (low variance in mu = more evaluations)
    
    mu_range = np.max(mu) - np.min(mu)
    mu_mean = np.mean(mu)
    
    # If mu is very concentrated, we've likely used more of our budget
    # Normalize: if mu_range is small relative to typical scale, we're late in optimization
    # Use a smooth surrogate for the "budget used" estimate
    
    # The cost of evaluating a point is 1 + mean(x_scaled), so costs are in [1, 2]
    # Mean cost of candidates gives us typical cost
    mean_candidate_cost = np.mean(cost_safe)
    
    # Estimate remaining budget: start with total budget 30
    # Subtract an estimate of used budget
    # Used budget ≈ (number of evaluations) * (average cost per evaluation)
    # We don't know exact number, but we can infer from GP uncertainty
    # Low sigma everywhere → many evaluations → less budget remaining
    
    sigma_mean = np.mean(sigma)
    sigma_max = np.max(sigma)
    
    # If sigma_mean is very low relative to sigma_max, we've explored a lot
    # Use a soft indicator: how "mature" is the GP?
    if sigma_max > 1e-9:
        maturity = 1.0 - (sigma_mean / sigma_max)
    else:
        maturity = 1.0
    
    # Clamp maturity to [0, 1]
    maturity = np.clip(maturity, 0.0, 1.0)
    
    # Estimate used budget: assume total budget is 30
    # Early on (maturity near 0), we've used little budget
    # Late on (maturity near 1), we've used most of our budget
    # This is a rough heuristic; the key is that remaining_budget decreases with maturity
    
    total_budget = 30.0
    # Estimate used budget as a function of maturity
    # Use a smooth curve: used_budget ≈ total_budget * maturity^0.5 (concave)
    used_budget_estimate = total_budget * np.sqrt(maturity)
    remaining_budget = max(total_budget - used_budget_estimate, 1.0)
    
    # 3. Cost-Effective Improvement: EI per unit of remaining budget
    # This is the core CEI metric
    # CEI = EI / (cost * remaining_budget_factor)
    
    # The cost of evaluating a specific candidate is cost[i]
    # We want to maximize improvement per unit cost, adjusted for remaining budget
    # As remaining budget shrinks, we should be more selective (only high-EI points)
    
    # CEI score = EI / (cost * remaining_budget)
    # But we need to handle the case where remaining_budget is small carefully
    
    cei_raw = ei / (cost_safe * remaining_budget)
    
    # 4. Add an exploration component that also respects cost
    # When EI is low (we're close to optimum), we need exploration
    # Use UCB-like term: (kappa * sigma) / (cost * remaining_budget)
    
    # Adaptive kappa: increase exploration when budget is tight
    # When remaining_budget is small, we should explore more aggressively
    # to find good points before running out of budget
    if remaining_budget < 5.0:
        kappa = 3.0  # More aggressive exploration when budget is tight
    elif remaining_budget < 15.0:
        kappa = 2.0
    else:
        kappa = 1.5  # Less aggressive exploration when budget is abundant
    
    exploration_term = kappa * sigma / (cost_safe * remaining_budget)
    
    # 5. Combine CEI and exploration
    # Normalize both to comparable scales
    cei_min, cei_max = np.min(cei_raw), np.max(cei_raw)
    exp_min, exp_max = np.min(exploration_term), np.max(exploration_term)
    
    cei_range = cei_max - cei_min
    exp_range = exp_max - exp_min
    
    if cei_range < 1e-12:
        cei_norm = np.zeros_like(cei_raw)
    else:
        cei_norm = (cei_raw - cei_min) / cei_range
    
    if exp_range < 1e-12:
        exp_norm = np.zeros_like(exploration_term)
    else:
        exp_norm = (exploration_term - exp_min) / exp_range
    
    # 6. Adaptive weighting based on maturity
    # Early on: more exploration (we need to map the landscape)
    # Late on: more exploitation (we need to refine the best point)
    # But also: if budget is tight, favor exploitation to lock in gains
    
    # Base weights
    base_explore_weight = 0.4  # Start with 40% exploration
    base_exploit_weight = 0.6  # Start with 60% exploitation
    
    # Adjust based on maturity
    # As maturity increases, shift weight toward exploitation
    explore_weight = base_explore_weight * (1.0 - 0.5 * maturity)
    exploit_weight = base_exploit_weight + 0.5 * base_explore_weight * maturity
    
    # Normalize weights to sum to 1
    weight_sum = explore_weight + exploit_weight
    explore_weight /= weight_sum
    exploit_weight /= weight_sum
    
    # 7. Final score
    score = exploit_weight * cei_norm + explore_weight * exp_norm
    
    # Ensure all values are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
