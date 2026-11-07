"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    """
    Computes attractiveness scores using Lagrangian Relaxation.
    
    Strategy:
    1. Estimate the 'shadow price' (dual variable) for each constraint.
       Since we cannot run an iterative solver easily within the strict time limits 
       for every single call if it's heavy, we use a fast heuristic approximation 
       of the dual variables based on the ratio of total weight to capacity.
       
    2. The heuristic value for an item is its total prize minus the weighted sum 
       of its weights, where the weights are the estimated shadow prices.
       
    3. Items with high prize and low "economic cost" (considering scarcity) will 
       have high scores.
    """
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    if prizes.shape != (5, 100) or weights.shape != (5, 100):
        # Fallback to simple ratio if shape is wrong (though contract says shape is fixed)
        total_prizes = np.sum(prizes, axis=0)
        total_weights = np.sum(weights, axis=0)
        heuristic = total_prizes / (total_weights + 1e-9)
        return np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)

    # 1. Estimate Shadow Prices (Dual Variables)
    # The capacity for each constraint is 25.
    capacity = 25.0
    
    # Total weight per constraint
    total_weight_per_constraint = np.sum(weights, axis=1) # shape (5,)
    
    # If a constraint is not fully utilized by the "average" item, its shadow price is low.
    # If it is very tight (total weight >> capacity), its shadow price is high.
    # We approximate the dual variable lambda_j as proportional to the "tightness".
    # A common heuristic for MCKP is lambda_j = C * (TotalWeight_j / Capacity_j)
    # Let's refine this: We want to penalize items that consume resources where the 
    # total demand exceeds supply significantly.
    
    # Utilization ratio: How many times the capacity is "used up" by all items combined?
    # Actually, for a single ant, it picks a subset. But for the heuristic, 
    # we look at the global scarcity.
    
    # Simple dual estimation:
    # If sum(weights[:, j]) is close to capacity, the constraint is tight.
    # But wait, the capacity is 25. The sum of all 100 items' weights for constraint j 
    # is likely much larger than 25 (since weights are uniform [0,1], avg 0.5, sum ~50).
    # So all constraints are "tight" in the global sense.
    
    # A better heuristic for Lagrangian relaxation in this context:
    # We want to maximize sum(prize_i) subject to sum(weight_ij * x_i) <= 25.
    # The Lagrangian is L = sum_i (prize_i - sum_j lambda_j * weight_ij) * x_i + sum_j lambda_j * 25.
    # To find good lambdas, we can use a subgradient method, but that's iterative.
    # Instead, let's use a static approximation based on "scarcity index".
    
    # Scarcity Index: How "expensive" is 1 unit of weight for constraint j?
    # If we assume the ant will pick the top K items by some metric, 
    # the shadow price reflects the marginal value of the constraint.
    
    # Let's try a different angle: 
    # The "value density" of an item should be adjusted by the "pressure" of each constraint.
    # Pressure_j = Total Weight_j / Capacity.
    # If Pressure is high, the constraint is more binding.
    
    pressures = total_weight_per_constraint / capacity
    
    # Normalize pressures so they are comparable and positive.
    # We use the pressures directly as the multipliers for the weights.
    # However, if pressures are very large (e.g., 50/25 = 2), the penalty might be too strong.
    # Let's scale them.
    
    # Alternative: Use the ratio of item weight to the *average* item weight for that constraint?
    # No, that's what the current program does (normalized cost).
    
    # Let's implement a simple 1-step Subgradient Update for the dual variables.
    # Initial lambdas: 0
    # We want to find lambdas such that the optimal solution to the relaxed problem 
    # satisfies the primal constraints.
    
    # Since we can't iterate much, let's use a heuristic initialization for lambdas:
    # lambda_j = alpha * (TotalWeight_j / Capacity - 1) if TotalWeight_j > Capacity else 0
    # This penalizes constraints that are "over-subscribed" by the total pool.
    
    over_subscription = np.maximum(0, total_weight_per_constraint / capacity - 1.0)
    
    # Scale factor: The total prize is roughly 5 * 100 * 0.5 = 250.
    # We want the penalty to be significant but not overwhelming.
    # Let's set alpha such that the average penalty is comparable to the average prize.
    # Avg prize per item ~ 2.5.
    # Avg weight per item ~ 0.5.
    # If lambda ~ 1, penalty ~ 0.5.
    
    # Let's just use the over_subscription as a relative weight.
    # If a constraint has total weight 50 (cap 25), over_sub = 1.
    # If total weight 25 (cap 25), over_sub = 0.
    
    # Actually, a more robust heuristic for MCKP heuristics is:
    # h_i = sum_j prize_ij - sum_j lambda_j * weight_ij
    # where lambda_j is estimated as the "marginal value" of the constraint.
    
    # Let's use a simple proportional allocation of the "tightness".
    # Tightness_j = max(0, TotalWeight_j - Capacity)
    tightness = np.maximum(0, total_weight_per_constraint - capacity)
    
    # If all tightness is 0, lambdas are 0.
    total_tightness = np.sum(tightness)
    
    if total_tightness < 1e-9:
        # All constraints are loose (unlikely with cap 25 and 100 items)
        lambdas = np.zeros(5)
    else:
        # Distribute a total penalty budget proportional to tightness.
        # What is the total penalty budget? 
        # Let's assume the total "value" at risk is the sum of all prizes.
        # We want the sum of penalties to be a fraction of total prize.
        total_prizes_sum = np.sum(prizes)
        
        # Heuristic: The sum of (lambda_j * sum_i weight_ij) should be ~ total_prizes_sum * 0.5
        # sum_j lambda_j * total_weight_j = 0.5 * total_prizes_sum
        # If we set lambda_j = k * tightness_j:
        # k * sum_j tightness_j * total_weight_j = 0.5 * total_prizes_sum
        # k = 0.5 * total_prizes_sum / (sum_j tightness_j * total_weight_j)
        
        denom = np.sum(tightness * total_weight_per_constraint)
        if denom < 1e-9:
            lambdas = np.zeros(5)
        else:
            k = (0.5 * total_prizes_sum) / denom
            lambdas = k * tightness
    
    # Calculate heuristic scores
    # score_i = sum_j prize_ij - sum_j lambda_j * weight_ij
    # This is the "net value" after accounting for resource scarcity.
    
    # Vectorized calculation:
    # prizes: (5, 100), lambdas: (5,), weights: (5, 100)
    # penalty = sum_j (lambda_j * weights[j, i]) for each i
    penalty = np.sum(lambdas[:, np.newaxis] * weights, axis=0) # (100,)
    total_prizes_per_item = np.sum(prizes, axis=0) # (100,)
    
    heuristic = total_prizes_per_item - penalty
    
    # Ensure positivity and finiteness
    # If heuristic is negative, the item is "economically unattractive".
    # The ACO algorithm requires positive heuristics (or they become 1e-9).
    # We should shift the values so that the "best" items are clearly higher.
    # However, if we shift too much, we lose the discrimination of the Lagrangian.
    
    # Strategy: 
    # 1. Clip negatives to a small positive value? No, that destroys ranking.
    # 2. Add a constant to make all positive? 
    #    If we add max(0, -min(heuristic)), the relative differences are preserved.
    
    min_h = np.min(heuristic)
    if min_h < 0:
        heuristic = heuristic - min_h + 1e-6
    
    # Check for non-finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Safety check for time (though this is fast)
    if time.time() - start_time > 290:
         # Fallback to simple ratio if something went wrong or took too long
         total_prizes = np.sum(prizes, axis=0)
         total_weights = np.sum(weights, axis=0)
         heuristic = total_prizes / (total_weights + 1e-9)
         return np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
        
    return heuristic
# EVOLVE-BLOCK-END
