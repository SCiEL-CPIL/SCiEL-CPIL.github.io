import numpy as np
import time

# Global deadline guard
_start_time = time.time()
TIME_LIMIT = 110.0 # Leave 10s margin for 120s limit

def _lagrangian_fill_value(capacity, weights, values, exclude_idx):
    """
    Estimate the value of filling the knapsack with remaining items
    using a Lagrangian relaxation approach, excluding item at exclude_idx.
    """
    if capacity <= 0:
        return 0.0
    
    n = len(weights)
    if n == 0:
        return 0.0
        
    # Create indices of all items except exclude_idx
    idxs = np.arange(n)
    if exclude_idx >= 0:
        idxs = np.delete(idxs, exclude_idx)
        
    if len(idxs) == 0:
        return 0.0
        
    w_rem = weights[idxs]
    v_rem = values[idxs]
    
    # Filter feasible items for this sub-problem (weight <= capacity)
    feasible_mask = w_rem <= capacity
    w_f = w_rem[feasible_mask]
    v_f = v_rem[feasible_mask]
    
    if len(w_f) == 0:
        return 0.0
        
    # Binary search for lambda
    # If lambda is very small, we pick almost everything (or high value/weight).
    # If lambda is very large, we pick nothing.
    # We want to find lambda such that sum(w_i * 1[v_i - lambda*w_i > 0]) <= capacity
    # Actually, the condition for item i to be selected in the fractional/relaxed sense
    # is v_i > lambda * w_i.
    # Let's define score_i = v_i - lambda * w_i.
    # We sort by score_i descending.
    # We take items while sum(weights) <= capacity.
    
    # Initial bounds for lambda
    # Max possible lambda is max(v/w). Min is 0.
    with np.errstate(divide='ignore', invalid='ignore'):
        max_density = np.max(v_f / w_f)
    
    lo = 0.0
    hi = max_density
    
    # We perform binary search to find the lambda that approximately matches capacity
    # The number of iterations should be small enough for speed but precise enough.
    # 20 iterations gives ~1e-6 relative precision.
    for _ in range(20):
        mid = (lo + hi) / 2.0
        scores = v_f - mid * w_f
        # Sort indices by score descending
        order = np.argsort(scores)[::-1]
        
        total_w = 0.0
        total_v = 0.0
        # We simulate the selection. In the Lagrangian relaxation, items with positive score are "wanted".
        # However, we must respect capacity. The standard approach is to sort by score and take as many as fit.
        # This is equivalent to the greedy algorithm on the modified scores.
        
        # Optimization: We can just sum weights of all items with positive score first?
        # No, because if sum(weights of positive score items) > capacity, we need to stop.
        # If sum(weights of positive score items) < capacity, we might want to include some zero/negative score items?
        # No, in Lagrangian relaxation for knapsack, the "dual" solution suggests taking items where v_i > lambda w_i.
        # If the total weight of these items is <= capacity, that's the set.
        # If it's > capacity, we have a "fractional" item at the boundary in the continuous case,
        # but for 0/1, we just take the top K that fit.
        
        # Let's just take the top items by score until capacity is full.
        for idx in order:
            w = w_f[idx]
            if total_w + w <= capacity:
                total_w += w
                total_v += v_f[idx]
            else:
                break
                
        if total_w > capacity:
            # This shouldn't happen with the loop above, but just in case
            pass
        elif total_w < capacity:
            # If we have remaining capacity, it means all positive score items fit.
            # We could potentially add more items with lower scores if they fit?
            # The Lagrangian objective is sum(v_i) - lambda * sum(w_i).
            # If we have space, adding an item with v_i - lambda w_i > 0 increases the objective.
            # If v_i - lambda w_i < 0, it decreases the objective.
            # So we should only add items if their score is positive.
            # Our loop above already stops when it can't fit. But does it stop if score is negative?
            # No, it stops when it can't fit OR if we run out of items.
            # We should explicitly stop if score becomes negative, because adding a negative score item
            # reduces the Lagrangian value, even if it fits.
            # Let's refine the loop to break if score <= 0.
            pass
            
        # To be more efficient and correct regarding the "break on negative score" logic:
        # Let's re-evaluate the selection for this mid.
        # Actually, the simple "sort by score and take until full" is a valid heuristic for the sub-problem.
        # If total_w < capacity, we are done (no more positive score items that fit? Not necessarily, but if all remaining have negative score, we stop).
        
        # Let's check if total_w is close to capacity.
        # If total_w < capacity, it implies either no more items or all remaining items have negative scores (or don't fit).
        # If there are items with positive scores that don't fit, we can't take them.
        # If there are items with negative scores that fit, we shouldn't take them.
        
        # So the current logic "take until full" might take negative score items if they fit and we haven't filled capacity?
        # No, if we sort by score descending, and we have capacity left, we check the next item.
        # If its score is negative, we should stop.
        
        # Let's implement a quick check for the break condition based on score.
        
        # Recalculate total_w and total_v with the score break condition for accuracy of the binary search direction?
        # Actually, for the binary search to work, we just need a monotonic function of lambda.
        # f(lambda) = total weight of selected items.
        # As lambda increases, scores decrease, so we select fewer items (or lower weight items).
        # So f(lambda) is non-increasing.
        
        # If total_w > capacity, we need to increase lambda (make scores lower, select less).
        # If total_w < capacity, we need to decrease lambda (make scores higher, select more).
        
        if total_w > capacity:
            lo = mid
        else:
            hi = mid
            
    # After binary search, we have a good lambda.
    # Now we perform the final selection using this lambda.
    # We sort by score = v - lambda * w.
    # We take items in that order if they fit and have positive score.
    
    final_lambda = (lo + hi) / 2.0
    scores = v_f - final_lambda * w_f
    order = np.argsort(scores)[::-1]
    
    total_v = 0.0
    rem_cap = capacity
    
    for idx in order:
        w = w_f[idx]
        v = v_f[idx]
        score = scores[idx]
        
        if score <= 0:
            break
            
        if w <= rem_cap:
            total_v += v
            rem_cap -= w
        else:
            # Skip if it doesn't fit.
            # Should we try to find a smaller item later?
            # The greedy approach on scores usually just skips.
            # Since we are in a heuristic, skipping is standard.
            pass
            
    return total_v

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    global _start_time
    
    # Time check
    if time.time() - _start_time > TIME_LIMIT:
        return -1
        
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1
        
    weights = np.asarray(remaining_weights, dtype=float)
    values = np.asarray(remaining_values, dtype=float)
    
    # Identify feasible items
    feasible_mask = weights <= remaining_capacity
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return -1
        
    best_val = -1.0
    best_idx = -1
    
    # Evaluate each feasible item as the next pick
    # Optimization: If we have many items, we might want to limit candidates to top K by density
    # to save time, but N=100 is small. Let's evaluate all feasible.
    
    for i in feasible_indices:
        w_i = weights[i]
        v_i = values[i]
        
        # Check time again inside loop if N was large, but N=100 is fine.
        # if time.time() - _start_time > TIME_LIMIT:
        #     break
        
        new_cap = remaining_capacity - w_i
        fill_val = _lagrangian_fill_value(new_cap, weights, values, i)
        total_est = v_i + fill_val
        
        if total_est > best_val:
            best_val = total_est
            best_idx = i
            
    return int(best_idx)
