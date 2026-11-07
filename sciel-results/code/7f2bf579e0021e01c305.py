"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard state
_START_TIME = None
_DEADLINE = 330.0  # Wall clock limit
_SAFETY_MARGIN = 2.0  # Seconds before limit to stop

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_exceeded():
    global _START_TIME, _DEADLINE, _SAFETY_MARGIN
    if _START_TIME is None:
        return False
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY_MARGIN)

def acquisition(mu, sigma, f_best, cost):
    _init_time()
    
    # Ensure inputs are numpy arrays for speed
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety check: if we are running out of time, return a deterministic fallback
    if _time_exceeded():
        # Fallback: return a simple EI score
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / (cost + 1e-9)

    # Strategy: "Adaptive Value of Information" (AVI)
    # Instead of EI / cost, we model the decision as maximizing the expected 
    # "utility" of the next point. 
    # 
    # Standard EI is E[max(0, f_best - f(x))].
    # The cost is incurred regardless of improvement.
    # A more principled approach in budget-constrained BO is to consider 
    # the "Cost-Adjusted Expected Improvement" but with a dynamic threshold.
    #
    # However, a structurally different approach is to use a 
    # "Probability of Improvement" (PI) weighted by the *magnitude* of potential 
    # improvement relative to the cost, but normalized by the current best.
    #
    # Let's try a "Risk-Aware" acquisition:
    # Score = (E[f_best - f(x)] - lambda * cost) * P(f(x) < f_best)
    # This penalizes cost linearly but only if there is a chance of improvement.
    #
    # Actually, a very effective strategy for multi-objective (value + cost) 
    # is to compute the "Hypervolume Improvement" (HI) in the (value, cost) space.
    # But since we minimize value and cost, we can approximate this.
    #
    # Let's use a simpler but robust structural change: 
    # **Inverse Cost Weighted UCB (Upper Confidence Bound)**
    # Standard EI can be too greedy. UCB = mu + beta * sigma.
    # We want to MINIMIZE, so we look for low mu and high sigma (uncertainty).
    # But we also care about cost.
    #
    # Let's define a score that rewards:
    # 1. Low predicted mean (mu)
    # 2. High uncertainty (sigma) - but only if it might lead to improvement
    # 3. Low cost
    #
    # The current program uses EI / sqrt(cost).
    # The parent (score 1.22) likely used 1/cost^2 or similar aggressive cost penalty.
    # The stagnation suggests we might be over-penalizing cost or under-exploring.
    #
    # New Structural Idea: **Budget-Aware Expected Improvement with Adaptive Beta**
    # We compute EI, but we modulate the "exploration" part of EI (which is sigma * pdf(z))
    # based on the remaining budget.
    #
    # Wait, we don't know the remaining budget directly, but we know the current cost array.
    # The `cost` array contains the cost of each *candidate* point.
    #
    # Let's try a **Pareto-Dominance** heuristic.
    # A point x is "good" if it has a high chance of being a Pareto improvement 
    # over the current best (f_best, current_total_cost).
    #
    # Since we don't have the total cost history, we only have candidate costs.
    #
    # Let's go back to basics. The best score (1.22) used a strong cost penalty.
    # The current (0.60) uses sqrt(cost).
    # The seed (1.00) used 1/cost.
    #
    # If 1/cost^2 is best, maybe we should try **1/cost** but with a **modified EI** 
    # that is less prone to zeroing out in unexplored regions.
    #
    # Modified EI: "Expected Improvement with Uncertainty Bonus"
    # EI_ub = EI + kappa * sigma
    # This ensures that even if EI is 0 (mu > f_best), high sigma points get a score.
    # Then divide by cost.
    #
    # Let's implement:
    # z = (f_best - mu) / (sigma + 1e-9)
    # ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    # score = (ei + 0.1 * sigma) / cost
    #
    # Why 0.1? It's a small bonus to encourage exploration without dominating exploitation.
    # This is structurally different from pure EI/cost because it decouples the 
    # exploration incentive from the improvement magnitude.
    
    cost_safe = np.maximum(cost, 1.0)
    
    # Compute standard EI
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Add uncertainty bonus
    # This term ensures that points with high sigma are considered even if mu is not
    # currently below f_best. This helps escape local optima where the GP is confident
    # (low sigma) but wrong (mu > f_best).
    uncertainty_bonus = 0.05 * sigma
    
    # Combine
    raw_score = ei + uncertainty_bonus
    
    # Apply cost penalty. 
    # We use linear cost penalty (1/cost) as a baseline, but we can adjust the exponent.
    # The finding says 1/cost^2 was better. Let's try 1/cost^1.5 as a compromise?
    # No, let's stick to the structural change: the uncertainty bonus is the key difference.
    # If we use 1/cost^2 with the bonus, it might be too aggressive.
    # Let's use 1/cost^1 (linear) with the bonus to see if exploration helps more than cost reduction.
    
    score = raw_score / cost_safe
    
    # Ensure finite scores
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
