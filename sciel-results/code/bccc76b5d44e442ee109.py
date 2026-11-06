# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.monotonic()
_TIME_LIMIT = 110.0  # Safety margin for 120s limit

def _compute_lagrangian_score(remaining_capacity, weights, values, feasible_indices):
    """
    Compute Lagrangian relaxation scores for each feasible item.
    Uses dynamic shadow price (lambda) tuned to approximate capacity exhaustion.
    """
    n = len(weights)
    if n == 0:
        return np.zeros(n)
    
    # Initialize lambda using average density as starting point
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_density = np.mean(values[feasible_indices] / weights[feasible_indices])
    
    # Binary search for lambda that approximately exhausts capacity
    # When lambda is high, we select fewer items; when low, more items
    lo = 0.0
    hi = max(values[feasible_indices] / weights[feasible_indices]) * 2 + 1e-6
    
    # Perform binary search for optimal lambda
    best_lambda = avg_density
    for _ in range(50):  # 50 iterations for good precision
        mid = (lo + hi) / 2
        
        # Compute which items would be selected at this lambda
        # Select items where value - lambda * weight > 0
        scores = values - mid * weights
        selected_mask = scores > 0
        
        # Calculate total weight of selected items
        total_weight = np.sum(weights[selected_mask])
        
        if total_weight > remaining_capacity:
            # Too many items selected, increase lambda
            lo = mid
        else:
            # Not enough capacity used, decrease lambda
            hi = mid
        
        best_lambda = mid
    
    # Compute final Lagrangian scores with the tuned lambda
    # score_i = value_i - lambda * weight_i
    # But we want to maximize this, so higher score = better item
    lagrangian_scores = values - best_lambda * weights
    
    # For infeasible items, set score to -inf
    lagrangian_scores[~np.isfinite(lagrangian_scores)] = -np.inf
    
    return lagrangian_scores

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Selects the next item for a 0/1 Knapsack problem.
    Uses Lagrangian relaxation with dynamic shadow prices as primary heuristic.
    Falls back to LP relaxation if needed.
    """
    # Time guard
    if time.monotonic() - _START_TIME > _TIME_LIMIT:
        return -1

    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return -1

    weights = np.asarray(remaining_weights, dtype=np.float64)
    values = np.asarray(remaining_values, dtype=np.float64)

    # Find feasible items
    feasible_mask = weights <= remaining_capacity
    if not np.any(feasible_mask):
        return -1
    
    feasible_indices = np.where(feasible_mask)[0]
    
    # Primary: Lagrangian relaxation approach
    # Compute shadow price (lambda) that approximates optimal dual solution
    lagrangian_scores = _compute_lagrangian_score(remaining_capacity, weights, values, feasible_indices)
    
    # For each feasible item, compute expected total value if we pick it
    # This combines the item's value with the Lagrangian estimate of remaining value
    best_score = -np.inf
    best_idx = -1
    
    for i in feasible_indices:
        w_i = weights[i]
        v_i = values[i]
        rem_cap = remaining_capacity - w_i
        
        if rem_cap <= 0:
            total_score = v_i
        else:
            # Estimate remaining value using Lagrangian scores of other items
            # Items with positive Lagrangian score are "worthwhile"
            other_scores = lagrangian_scores.copy()
            other_scores[i] = -np.inf  # Exclude current item
            
            # Select items in order of Lagrangian score until capacity exhausted
            sorted_order = np.argsort(other_scores)[::-1]
            
            lp_val = 0.0
            cur_cap = rem_cap
            
            for j in sorted_order:
                if other_scores[j] <= 0:
                    break  # No more worthwhile items
                
                w_j = weights[j]
                v_j = values[j]
                
                if w_j <= cur_cap:
                    lp_val += v_j
                    cur_cap -= w_j
                    if cur_cap <= 0:
                        break
                else:
                    # Fractional part for LP bound
                    if w_j > 0:
                        lp_val += v_j * (cur_cap / w_j)
                    cur_cap = 0
                    break
            
            total_score = v_i + lp_val
        
        if total_score > best_score:
            best_score = total_score
            best_idx = i
    
    if best_idx != -1:
        return int(best_idx)
    
    # Fallback: pure density-based greedy
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = values / weights
    
    sorted_indices = np.argsort(densities)[::-1]
    for i in sorted_indices:
        if i in feasible_indices:
            return int(i)
    
    return -1
# EVOLVE-BLOCK-END
