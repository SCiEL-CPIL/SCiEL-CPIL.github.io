"""
Budget-Aware Expected Improvement (BA-EI) with dynamic cost filtering.

Strategy:
Treats the BO loop as a budget-constrained sequential decision problem.
The acquisition function adapts to the remaining budget:
1. Calculate remaining budget fraction r = (Budget - Current_Cost) / Budget.
2. Adjust exploration coefficient beta = beta_max * sqrt(r).
   - When r is high (start), beta is high -> explore.
   - When r is low (end), beta is low -> exploit.
3. Apply a cost-efficiency mask:
   - Estimate average remaining cost per step.
   - If a candidate's cost is significantly higher than the average remaining cost 
     AND the expected improvement is low relative to the cost, penalize it heavily.
   - Specifically, we compute a 'cost-adjusted EI' where the cost acts as a 
     dynamic threshold.
   
   To keep it simple and robust without knowing the exact number of steps left, 
   we assume a fixed total budget of 30.0 as per the task description.
   We estimate the 'value per cost' and compare it to a threshold that tightens 
   as the budget depletes.

   Score = EI * (1 + beta * sigma / EI_safe) / (1 + (cost / cost_ref)^gamma)
   
   Where cost_ref is the average cost of the candidates, and gamma increases 
   as budget decreases.
"""
import numpy as np
from scipy.stats import norm
import time

# EVOLVE-BLOCK-START

# Global state to track time if needed, but primarily we rely on input 'cost'
# The task says 'cost' has shape (2000,), which is the cost of evaluating each candidate.
# f_best is the current best value.
# We need to infer the 'remaining budget'.
# The task states: "While total cost is below 30".
# It does NOT pass the current total cost to the acquisition function.
# However, we can infer the current step or approximate the remaining budget 
# if we track state. But the function signature is stateless.
# Wait, the task says: "mu, sigma and cost have shape (2000,)."
# It does NOT pass the current total cost.
# This is a critical constraint. We cannot know the exact remaining budget 
# unless we track it in module-level state.
# Let's use a module-level variable to track the cumulative cost of evaluations.
# We update this variable every time acquisition is called? 
# No, acquisition is called *before* evaluation. 
# The loop is:
# 1. Generate 2000 candidates.
# 2. Call acquisition(mu, sigma, f_best, cost).
# 3. Select best candidate.
# 4. Evaluate it. Add its cost to total_cost.
# 5. Update GP.
# 6. Repeat.
#
# Since we don't receive the current total cost, we must track it.
# We can estimate the number of steps taken by counting calls to acquisition.
# However, the first 5 points are "uniform random points". 
# The acquisition function is only called for the *subsequent* 2000 candidates.
# So, call count = number of BO steps performed.
# Initial cost is sum of costs of 5 random points. We don't know those costs.
# But the cost function is 1 + mean(x_scaled). 
# Mean cost of a uniform point in [0,1]^2 is 1 + 0.5 = 1.5.
# So initial cost is approx 5 * 1.5 = 7.5.
# We can track the cumulative cost of the *selected* points in our state.
# We need to know which point was selected. 
# The acquisition function returns scores. The evaluator picks the max.
# We can store the selected cost in our state.

_total_cost = 7.5  # Approximate initial cost
_num_calls = 0     # Number of acquisition calls (BO steps)
_MAX_BUDGET = 30.0

def acquisition(mu, sigma, f_best, cost):
    global _total_cost, _num_calls
    
    # Update state: 
    # We don't know which point was picked in the *previous* call.
    # But we can assume the evaluator picks the argmax of our returned scores.
    # We can store the max cost from the previous call's input? 
    # No, that's not the selected cost.
    # Actually, we can just track the number of steps and assume an average cost.
    # Or, better: We can estimate the remaining budget based on the number of calls.
    # If we've made N calls, we've spent some cost.
    # Let's just use the number of calls as a proxy for budget depletion.
    # Total budget 30. Initial ~7.5. Remaining ~22.5.
    # If we assume average cost per step is ~1.5 (conservative, as BO tends to 
    # find cheaper or similar cost points initially, but might explore expensive ones),
    # we have roughly 15 steps.
    # Let's track the actual selected cost if possible? 
    # We can't modify the evaluator.
    # However, we CAN update our internal estimate of _total_cost 
    # by adding the cost of the point we *would* have picked (argmax score).
    # This is a simulation of the evaluator's choice.
    
    # First, calculate scores to find the argmax (simulating the choice)
    # But we need the scores to return.
    # So:
    # 1. Calculate scores.
    # 2. Find index of max score.
    # 3. Update _total_cost with cost[idx].
    # 4. Increment _num_calls.
    # 5. Return scores.
    
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Calculate EI
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Estimate remaining budget
    # We track _total_cost.
    # remaining = _MAX_BUDGET - _total_cost
    # If remaining <= 0, we should just exploit the best mu (low cost).
    
    remaining_budget = _MAX_BUDGET - _total_cost
    if remaining_budget < 0:
        remaining_budget = 0.0
    
    # Budget fraction
    # We assume total available budget for BO steps is ~22.5 (30 - 7.5)
    # Let's define a reference budget for normalization
    REF_BUDGET = 22.5
    budget_ratio = np.clip(remaining_budget / REF_BUDGET, 0.0, 1.0)
    
    # Dynamic Beta
    # High beta when budget is high (explore), low beta when budget is low (exploit)
    # Beta = 2.0 * budget_ratio + 0.1
    beta = 2.0 * budget_ratio + 0.1
    
    # UCB-like term for exploration
    ucb_term = beta * sigma_safe
    
    # Combine EI and UCB
    # Standard EI is good for exploitation. UCB is good for exploration.
    # We want to blend them.
    # Score = EI + UCB_term
    # But we need cost awareness.
    
    # Cost Awareness Strategy:
    # Penalize high cost points more when budget is low.
    # Let's use a cost penalty that scales with (1 - budget_ratio).
    # penalty_factor = 1.0 + (1.0 - budget_ratio) * 5.0 * (cost / mean_cost)
    # This makes expensive points much less attractive when budget is low.
    
    mean_cost = np.mean(cost)
    if mean_cost == 0:
        mean_cost = 1.0
    
    # Normalize cost to [0, 1] roughly
    cost_norm = cost / (mean_cost * 2.0) # Scale so mean is ~0.5
    
    # Penalty strength increases as budget decreases
    # strength = 1.0 / (budget_ratio + 0.1)
    # If budget_ratio is 1, strength ~ 1.
    # If budget_ratio is 0, strength ~ 10.
    strength = 1.0 / (budget_ratio + 0.1)
    
    # Cost penalty: multiply score by 1 / (1 + strength * cost_norm^2)
    # This is a soft filter.
    penalty = 1.0 / (1.0 + strength * np.power(cost_norm, 2.0))
    
    # Final Score
    score = (ei + ucb_term) * penalty
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Simulate evaluator choice to update state
    # The evaluator picks the candidate with the highest score.
    idx_max = np.argmax(score)
    _total_cost += cost[idx_max]
    _num_calls += 1
    
    return score

# EVOLVE-BLOCK-END
