import numpy as np

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    n = len(remaining_weights)
    if n == 0:
        return -1
        
    W = np.array(remaining_weights)
    V = np.array(remaining_values)
    C = remaining_capacity
    
    # Check for feasible items
    feasible_mask = W <= C
    if not np.any(feasible_mask):
        return -1
        
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 1:
        return int(feasible_indices[0])

    # Extract weights and values for feasible items
    W_f = W[feasible_indices]
    V_f = V[feasible_indices]
    
    # Calculate densities
    # Avoid division by zero (weights should be > 0, but just in case)
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = V_f / W_f
    
    # Handle zero weights (should not happen in standard knapsack, but if it does, density is inf)
    # If w==0 and v>0, it's always good.
    # If w==0 and v==0, it's neutral.
    
    # Find the critical density (lambda)
    # Sort by density descending
    # np.nan handling? If W has 0, density is inf. Inf sorts first.
    sort_idx = np.argsort(densities)[::-1]
    W_sorted = W_f[sort_idx]
    V_sorted = V_f[sort_idx]
    
    cumulative_weight = 0.0
    lambda_val = 0.0
    
    for w, v in zip(W_sorted, V_sorted):
        if cumulative_weight + w <= C:
            cumulative_weight += w
        else:
            if w > 0:
                lambda_val = v / w
            else:
                # If w==0, it fits. This case shouldn't hit the 'else' if w=0 fits in remaining capacity?
                # If w=0, cumulative_weight + 0 <= C is always true if C>=0.
                # So w=0 items are always added to cumulative_weight without breaking the loop.
                # Thus, lambda_val will be set by a w>0 item if any exist that cause overflow.
                pass
            break
            
    # If we didn't break, all items fit. lambda_val = 0.
    
    # Calculate Lagrangian scores: v_i - lambda * w_i
    scores = V_f - lambda_val * W_f
    
    # We want the item with the maximum score.
    # To break ties deterministically and perhaps favor "better" items, 
    # we can use a composite key or just rely on argmax.
    # Let's add a small epsilon based on density to break ties in favor of higher density?
    # Or just pick the first one.
    
    # A common refinement: if scores are very close, prefer the item with higher density 
    # because it leaves more "value per weight" potential for the rest? 
    # Actually, if scores are equal, $v_1 - \lambda w_1 = v_2 - \lambda w_2$.
    # $v_1 = v_2 + \lambda(w_1 - w_2)$.
    # If $w_1 > w_2$, then $v_1 > v_2$.
    # Picking the heavier one consumes more capacity.
    # It's often safer to pick the lighter one if scores are equal?
    # Or the one with higher density?
    
    # Let's stick to the primary signal: Max Score.
    # If there are multiple max scores, pick the one with highest density.
    
    max_score = np.max(scores)
    # Find indices with max score (with tolerance)
    # Using np.isclose for floating point
    candidates = np.where(np.isclose(scores, max_score))[0]
    
    if len(candidates) == 1:
        best_local_idx = candidates[0]
    else:
        # Among candidates, pick the one with highest density
        cand_densities = densities[candidates]
        best_local_idx = candidates[np.argmax(cand_densities)]
        
    # Map back to original index
    best_global_idx = feasible_indices[best_local_idx]
    
    return int(best_global_idx)
