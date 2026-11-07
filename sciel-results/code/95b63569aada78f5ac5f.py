import numpy as np
import time

def heuristics(prizes, weights):
    start_time = time.time()
    
    # prizes: (5, 100)
    # weights: (5, 100)
    
    # 1. Calculate total prize per item (sum across constraints? No, prizes are likely per-item values 
    #    but the shape is (5, 100). Usually in MCKP, prize is a single vector (100,) or matrix (5,100) 
    #    if it's a multi-objective. The prompt says "mean over instances of the highest total prize".
    #    "total prize" usually implies summing the prize vector if it's multi-dimensional, 
    #    or if prizes is (5,100), it might be 5 different prize vectors? 
    #    Wait, the prompt says: "prizes and weights, of shape (5, 100)".
    #    "highest total prize found by any ant".
    #    If an ant picks a set of items S, the total prize is likely sum_{i in S} sum_{c} prizes[c, i] 
    #    or just sum_{i in S} prizes[i]? 
    #    Given the shape (5, 100), it's ambiguous. However, standard MCKP has one prize per item.
    #    If prizes is (5, 100), maybe it's 5 different prize coefficients? 
    #    Let's assume "total prize" means the sum of all entries in the prize vector for the selected items.
    #    So item i has value V_i = sum_{c=0}^{4} prizes[c, i].
    
    total_prize_per_item = np.sum(prizes, axis=0)
    
    # 2. Estimate Shadow Prices (Lagrange Multipliers)
    # The capacity for each constraint is 25.
    # Total available resource for constraint c is 25.
    # Total demand for constraint c is sum(weights[c, :]).
    # If demand > supply, the resource is scarce.
    # A simple proxy for shadow price lambda_c is proportional to the scarcity.
    # If sum(weights) > 25, lambda_c > 0.
    # Let's define scarcity_c = max(0, sum(weights[c, :]) - 25) / 25.
    # Or simply lambda_c = 1 / (capacity / total_weight) if total_weight > 0?
    # The prompt suggests: "inverse of its total weight sum".
    # Let's use lambda_c = 1 / sum(weights[c, :]) if sum > 0 else 0.
    # This makes constraints with high total weight "cheap" (low shadow price) in this specific heuristic?
    # No, usually higher scarcity -> higher price.
    # If total weight is high, the constraint is tight.
    # Let's look at the suggested strategy: "inverse of its total weight sum".
    # If sum(weights) is large, 1/sum is small. This would imply tight constraints have low prices?
    # That seems counter-intuitive for a penalty.
    # Let's re-read: "estimating the 'shadow price' of each constraint via the inverse of its total weight sum".
    # Maybe it means the price is proportional to scarcity, and scarcity is related to 1/weight?
    # Actually, if we just follow the prompt's literal suggestion for the *mechanism*:
    # "scoring items by their prize minus the weighted sum of their resource costs"
    # Cost_i = sum_c (lambda_c * weights[c, i])
    # Score_i = total_prize_per_item[i] - Cost_i
    
    # Let's refine lambda_c.
    # If we use lambda_c = 1 / sum(weights[c, :]), then items in constraints with high total weight
    # get penalized less. This might be wrong if high total weight means high competition.
    # However, the prompt explicitly suggests this direction. Let's try a variation that makes more sense:
    # Shadow price should be high when the constraint is tight.
    # Tightness = total_weight / capacity.
    # If total_weight > capacity, it's infeasible to take all.
    # Let's use lambda_c = max(0, total_weight_c - capacity) / capacity.
    # This is 0 if not tight, positive if tight.
    
    capacities = np.full(5, 25.0)
    total_weights = np.sum(weights, axis=1)
    
    # Scarcity factor: how much over capacity are we if we took everything?
    # If total_weights[c] > 25, the constraint is binding.
    # Let's use a normalized scarcity.
    scarcity = np.maximum(0, total_weights - capacities) / capacities
    
    # To avoid zero division or trivial cases, we can also incorporate the base inverse weight idea if needed,
    # but scarcity is the standard economic interpretation.
    # Let's stick to the prompt's "inverse of total weight sum" idea but adapted for penalty logic:
    # If the prompt implies using 1/sum(weights), maybe it's a heuristic for "efficiency"?
    # Let's try: lambda_c = 1.0 / (total_weights[c] + 1e-9)
    # Then Score = Prize - sum(lambda_c * weight_ci).
    # This penalizes items in constraints where the *total* weight is low (i.e., easy to satisfy? no, that's wrong).
    # If total weight is low, 1/sum is high. So items in "easy" constraints are penalized MORE?
    # That seems backwards.
    
    # Let's look at the inspiration 3 or general MCKP heuristics.
    # Usually, we want to pick items with high Prize / Weight ratio.
    # The Lagrangian relaxation of max sum(p_i x_i) s.t. sum(w_ci x_i) <= C_c
    # is max sum((p_i - sum_c lambda_c w_ci) x_i)
    # We want lambda_c to reflect the tightness.
    # A common simple estimate: lambda_c = (sum_all_w_ci - C_c) / C_c if positive, else 0.
    # Or simply lambda_c = 1 if tight, 0 if not.
    
    # Let's try the "Scarcity" approach:
    # lambda_c = max(0, total_weights[c] - capacities[c])
    # This is the amount of "excess" demand.
    
    # Calculate the weighted cost for each item
    # weights is (5, 100), scarcity is (5,)
    # cost_matrix = weights * scarcity[:, np.newaxis] -> (5, 100)
    # total_cost_per_item = sum(cost_matrix, axis=0) -> (100,)
    
    excess_demand = np.maximum(0, total_weights - capacities)
    
    # Normalize the excess demand to be comparable across constraints?
    # If one constraint is 10 units over and another is 1 unit over, the first is more critical.
    # Let's use the raw excess demand as the shadow price.
    
    shadow_prices = excess_demand
    
    # Calculate the "net value" for each item
    # net_value_i = total_prize_i - sum_c (shadow_price_c * weight_ci)
    
    weighted_costs = np.sum(weights * shadow_prices[:, np.newaxis], axis=0)
    net_values = total_prize_per_item - weighted_costs
    
    # The heuristic in ACO is usually positive.
    # If net_values are negative, the ant should avoid them, but the heuristic must be positive.
    # The prompt says: "Heuristic entries that are not positive become 1e-9".
    # So we can just return net_values, and the ACO logic will handle the negatives by clamping to 1e-9.
    # However, a better heuristic might be to shift/scale them to be all positive and preserve relative order.
    # If we just return net_values, items with negative net value get 1e-9 (very low probability).
    # Items with positive net value get their value.
    # This seems reasonable.
    
    eta = net_values
    
    # Ensure positivity and finite values as per prompt requirements
    # The prompt says: "Heuristic entries that are not positive become 1e-9"
    # This implies the function can return non-positive values, and the *caller* (or the ACO loop) handles it?
    # "Heuristic entries that are not positive become 1e-9" is in the description of the ACO process.
    # "Write heuristics... returns one attractiveness score... (larger is more attractive)"
    # "An exception, non-finite values or a wrong shape make the whole result invalid."
    # It does NOT say the function must return positive values. It says the ACO process clamps them.
    # BUT, to be safe and ensure "larger is more attractive" works well without relying on the clamp
    # for the primary ranking (since 1e-9 is tiny), it's better to return values that are naturally positive
    # if possible, or at least correctly ordered.
    
    # If net_values are all negative, all become 1e-9, and the choice is random (or based on pheromone).
    # This might be okay if all items are "bad".
    
    # Let's add a small epsilon to ensure no NaNs/Infs
    eta = np.where(np.isfinite(eta), eta, 1e-9)
    
    # Deadline guard
    if time.time() - start_time > 290:
        return eta
        
    return eta
