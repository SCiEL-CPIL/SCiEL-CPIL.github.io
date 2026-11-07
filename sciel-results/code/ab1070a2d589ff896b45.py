# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for each item using a dynamic marginal utility heuristic.
    
    The heuristic estimates the opportunity cost of each item by considering the scarcity
    of each resource (constraint). Items that consume scarce resources are penalized more,
    while items that offer high prize relative to the scarcity of resources they use are rewarded.
    
    Approach:
    1. For each constraint (resource), compute the "efficiency" of each item as prize/weight.
    2. For each constraint, identify the top-k most efficient items (those that get the most
       prize per unit of that resource). These represent the "opportunity cost" of using that resource.
    3. Compute a scarcity score for each constraint based on how much of the capacity is "claimed"
       by the top efficient items.
    4. For each item, compute a weighted penalty based on how much it consumes from scarce resources.
    5. The final heuristic is prize / (1 + weighted_scarcity_penalty), which rewards items with
       high prize and low consumption of scarce resources.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    n_constraints, n_items = prizes.shape
    capacity = 25.0
    
    # Clip weights to avoid division by zero
    weights_safe = np.clip(weights, 1e-9, None)
    
    # Compute efficiency for each item in each constraint: prize / weight
    # Shape: (n_constraints, n_items)
    efficiency = prizes / weights_safe
    
    # For each constraint, find the top-k most efficient items
    # k is chosen as roughly capacity / mean_weight to estimate how many items "fit"
    k_list = []
    for c in range(n_constraints):
        mean_w = np.mean(weights[c, :])
        if mean_w > 0:
            k = max(1, int(capacity / mean_w))
        else:
            k = 1
        k = min(k, n_items)
        k_list.append(k)
    
    # Compute scarcity for each constraint
    # For each constraint, the top-k items are the most efficient users of that resource.
    # The "scarcity" is how much of the capacity is consumed by these top items.
    # We use the sum of weights of top-k items as a proxy for scarcity.
    scarcity = np.zeros(n_constraints)
    for c in range(n_constraints):
        k = k_list[c]
        # Get the top-k items by efficiency for this constraint
        top_k_indices = np.argpartition(efficiency[c, :], -k)[-k:]
        top_k_weights = weights[c, top_k_indices]
        top_k_prizes = prizes[c, top_k_indices]
        
        # Scarcity: ratio of total weight of top-k items to capacity
        # If the top-k items already use a lot of the capacity, the resource is scarce
        total_weight_topk = np.sum(top_k_weights)
        scarcity[c] = total_weight_topk / capacity if capacity > 0 else 0.0
    
    # Normalize scarcity to [0, 1] range for better scaling
    max_scarcity = np.max(scarcity)
    if max_scarcity > 0:
        scarcity = scarcity / max_scarcity
    
    # For each item, compute the weighted scarcity penalty
    # The penalty is the sum over all constraints of: weight[c, i] * scarcity[c]
    # This represents how much "scarce resource" the item consumes
    # Shape: (n_items,)
    weighted_scarcity_penalty = np.zeros(n_items)
    for c in range(n_constraints):
        weighted_scarcity_penalty += weights[c, :] * scarcity[c]
    
    # The heuristic: prize / (1 + weighted_scarcity_penalty)
    # Items with high prize and low scarcity penalty get high scores
    total_prize_per_item = np.sum(prizes, axis=0)  # Sum of prizes across constraints
    heuristic = total_prize_per_item / (1.0 + weighted_scarcity_penalty + 1e-9)
    
    # Ensure all values are positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
