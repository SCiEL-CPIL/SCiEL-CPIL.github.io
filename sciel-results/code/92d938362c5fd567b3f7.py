"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    """
    Computes attractiveness scores using a Greedy Simulation Heuristic.
    
    Strategy:
    1. Simulate a greedy packing process to estimate the 'marginal utility' of each item.
    2. Instead of a static ratio, we simulate how items would be selected if we were
       filling the knapsack greedily based on immediate prize density.
    3. Items that are selected early in this greedy simulation (when resources are abundant)
       get a higher score, but we also adjust for items that are 'blocked' by constraints.
    
    Specifically, we run a simplified greedy loop:
    - Sort items by prize/weight ratio (or similar efficiency metric).
    - Iterate through them, adding to a virtual knapsack if they fit.
    - The heuristic score reflects the 'value' an item would have added at the moment
      it was considered, penalized by the 'scarcity' of the resources it consumes at that time.
    
    To make this robust and fast for 100 items x 5 constraints:
    - We use a proxy for scarcity: the remaining capacity of each constraint.
    - We define the heuristic for item i as:
        prize[i] / (1 + sum_j (weight[i,j] / remaining_capacity[j])^2)
      where remaining_capacity is estimated from a preliminary greedy pass.
      
    However, a simpler and often more effective structural change is to use the 
    'regret' or 'opportunity cost' derived from a greedy solution.
    
    Let's implement a 'Greedy Regret' heuristic:
    1. Compute a base efficiency score for each item: eff[i] = sum(prizes[:, i]) / sum(weights[:, i])
    2. Perform a greedy selection:
       - Start with empty knapsack (capacity 25 for all 5 constraints).
       - Repeatedly pick the item with highest eff[i] that still fits.
       - If an item doesn't fit, record the 'tightest' constraint blocking it.
    3. The heuristic score for an item is boosted if it was selected early, 
       and damped if it was frequently blocked by a specific tight constraint.
       
    Actually, the prompt asks for a heuristic that guides an Ant Colony Optimization.
    ACO heuristics are typically static per instance.
    
    Let's try a different angle: **Constraint-Aware Prize Density**.
    The current program uses a global normalization.
    A better structural change is to normalize weights by the *capacity* (which is fixed at 25) 
    and then compute a "pressure" score for each constraint based on the average weight.
    
    New Strategy: **Dynamic Scarcity Weighting**
    1. For each constraint c, calculate the average weight of all items: avg_w[c].
    2. The "scarcity" of constraint c is proportional to how much of the capacity is "used up" 
       by the average item. Since capacity is 25, and we have 100 items, if avg_w[c] is high, 
       that constraint is tighter.
    3. Compute a "shadow price" lambda[c] for each constraint. 
       A simple estimate: lambda[c] = 1 / (capacity - k * avg_w[c]) where k is an estimated 
       number of items that will fit. But we don't know k.
       
    Let's stick to the **Greedy Simulation** idea but make it a static score calculation 
    that mimics the dynamics.
    
    Algorithm:
    1. Calculate "raw efficiency" for each item: e[i] = sum(prizes[:, i]) / (sum(weights[:, i]) + eps)
    2. Sort items by e[i] descending.
    3. Simulate filling the knapsack:
       - current_weights = zeros(5)
       - capacity = 25.0
       - selected = []
       - for i in sorted_indices:
           if all(current_weights + weights[:, i] <= capacity):
               # Item i fits. 
               # The "value" of selecting i now is its prize, but the "cost" is 
               # the reduction in future flexibility.
               # We assign a score to i based on the "slack" remaining after adding it.
               # High slack -> good choice. Low slack -> risky choice (but maybe high prize).
               # Let's score i as: prize[i] * (slack_after_adding_i / capacity)
               # This rewards items that leave more room for others.
               
               new_weights = current_weights + weights[:, i]
               slacks = capacity - new_weights
               min_slack = np.min(slacks)
               # If min_slack is low, this item is "tight".
               # Heuristic score for i: prize[i] * (1 + min_slack/capacity)
               score[i] = np.sum(prizes[:, i]) * (1.0 + min_slack / 25.0)
               
               current_weights = new_weights
               selected.append(i)
           else:
               # Item i does not fit. 
               # It is blocked. We give it a lower score, but not zero, 
               # because ACO might still pick it if pheromones are high.
               # Score based on how close it was to fitting?
               # Or just a baseline efficiency score?
               score[i] = np.sum(prizes[:, i]) * 0.5 # Penalty for not fitting in greedy
               
    4. Normalize scores to be positive.
    
    This captures the "marginal utility" better than a static ratio because it accounts 
    for the interaction between items (an item that fits easily is more attractive 
    in the context of a greedy strategy than one that barely fits).
    """
    start_time = time.time()
    deadline = 300.0 # Safety margin
    
    n_constraints, n_items = prizes.shape
    capacity = 25.0
    eps = 1e-9
    
    # 1. Base efficiency for sorting
    total_prizes = np.sum(prizes, axis=0)
    total_weights = np.sum(weights, axis=0)
    
    # Avoid division by zero
    total_weights_safe = np.where(total_weights > eps, total_weights, 1.0)
    efficiency = total_prizes / total_weights_safe
    
    # Sort items by efficiency descending
    sorted_indices = np.argsort(-efficiency)
    
    scores = np.zeros(n_items)
    current_weights = np.zeros(n_constraints)
    
    # 2. Greedy Simulation
    for idx in sorted_indices:
        if time.time() - start_time > deadline - 5.0:
            # Fallback if taking too long (unlikely for 100 items)
            break
            
        item_weights = weights[:, idx]
        item_prize = total_prizes[idx]
        
        # Check if it fits
        if np.all(current_weights + item_weights <= capacity):
            # It fits. Calculate the "slack" after adding it.
            new_weights = current_weights + item_weights
            slacks = capacity - new_weights
            min_slack = np.min(slacks)
            
            # Heuristic: Prize scaled by remaining flexibility.
            # Items that leave more room are "better" for future selections.
            # This mimics a dynamic programming state value roughly.
            flexibility_factor = 1.0 + (min_slack / capacity)
            scores[idx] = item_prize * flexibility_factor
            
            # Update current weights
            current_weights = new_weights
        else:
            # It doesn't fit. 
            # Assign a score based on its raw efficiency, but penalized.
            # This ensures it's still considered, but less attractive.
            scores[idx] = item_prize * 0.8 # Slight penalty for being "blocked"
            
    # 3. Finalize scores
    # Ensure all are positive and finite
    scores = np.where(np.isfinite(scores) & (scores > 0), scores, 1e-9)
    
    # Optional: Normalize to [0, 1] or keep raw? 
    # The task says "larger is more attractive". 
    # ACO uses pheromone * heuristic. 
    # If heuristics are very large, they might dominate pheromones initially.
    # Let's normalize by the max score to keep them in a reasonable range [0, 1] or similar.
    max_score = np.max(scores)
    if max_score > 0:
        scores = scores / max_score
        
    # Ensure minimum value
    scores = np.where(scores > 0, scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
