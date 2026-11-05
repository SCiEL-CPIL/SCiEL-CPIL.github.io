import numpy as np
import time

# Global deadline guard
_start_time = time.time()
TIME_LIMIT = 110.0 # Leave 10s margin for 120s limit

def _fractional_knapsack_value(capacity, weights, values):
    """
    Calculate the upper bound value using the fractional knapsack heuristic.
    This is O(n log n) due to sorting, but we can optimize by passing pre-sorted lists.
    For simplicity and safety, we sort here.
    """
    if capacity <= 0 or len(weights) == 0:
        return 0.0
        
    # Create indices sorted by density (value/weight) descending
    # Handle weight 0 cases: if weight is 0 and value > 0, it's always included.
    # If weight is 0 and value <= 0, it's ignored.
    
    n = len(weights)
    # Filter out items with weight > capacity (they can't be added fully, 
    # but in fractional knapsack, we just skip them or take 0. 
    # Actually, standard fractional knapsack assumes we can take fractions.
    # Items with weight > capacity can still contribute if we take a fraction?
    # No, in the context of a bound for 0/1, the fractional bound assumes 
    # we can take any amount. So if w > cap, we can take cap/w fraction.
    
    # However, for the bound to be tight, we should consider all items.
    
    # To speed up, we can avoid full sort if we just need the top few, 
    # but n is small (<=100), so sort is fine.
    
    # Calculate densities
    with np.errstate(divide='ignore', invalid='ignore'):
        densities = np.where(weights > 0, values / weights, np.inf)
        
    # Sort indices by density descending
    sorted_indices = np.argsort(-densities)
    
    total_value = 0.0
    remaining_cap = capacity
    
    for idx in sorted_indices:
        if remaining_cap <= 0:
            break
            
        w = weights[idx]
        v = values[idx]
        
        if w <= 0:
            # Zero weight item
            if v > 0:
                total_value += v
            # else ignore
        else:
            if w <= remaining_cap:
                total_value += v
                remaining_cap -= w
            else:
                # Take fraction
                total_value += v * (remaining_cap / w)
                remaining_cap = 0
                
    return total_value

def _dp_solve_exact(remaining_capacity, remaining_weights, remaining_values):
    """
    Solves the 0/1 knapsack exactly using DP.
    Returns the list of indices of the optimal solution.
    """
    n = len(remaining_weights)
    if n == 0 or remaining_capacity <= 0:
        return []

    # Scale to integers for DP
    scale = 1000
    C_int = int(round(remaining_capacity * scale))
    
    # If capacity is too large for DP, return empty (caller should handle)
    if C_int > 50000:
        return None # Signal fallback
        
    w_ints = np.array([int(round(w * scale)) for w in remaining_weights])
    values = np.asarray(remaining_values, dtype=float)
    
    # DP table
    # dp[j] = max value achievable with capacity j
    # We use 1D DP for space efficiency, but need to backtrack.
    # For backtracking, we need the full 2D table or store decisions.
    # Given n is small (<20 in the hybrid strategy), 2D is fine.
    
    dp = np.zeros((n + 1, C_int + 1), dtype=float)
    
    for i in range(1, n + 1):
        w = w_ints[i - 1]
        v = values[i - 1]
        
        # Copy previous row
        dp[i, :] = dp[i - 1, :]
        
        if w > 0 and w <= C_int:
            # Update from right to left? No, we are using 2D.
            # dp[i, j] = max(dp[i-1, j], dp[i-1, j-w] + v)
            dp[i, w:] = np.maximum(dp[i, w:], dp[i - 1, :-w] + v)
        elif w == 0:
            if v > 0:
                dp[i, :] = dp[i - 1, :] + v
            # else dp[i, :] is already dp[i-1, :]
            
    # Backtrack
    solution = []
    j = C_int
    for i in range(n, 0, -1):
        if dp[i, j] != dp[i - 1, j]:
            solution.append(i - 1)
            w = w_ints[i - 1]
            if w > 0:
                j -= w
            # if w == 0, j doesn't change, but item is included
            
    return solution

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
    
    # Filter feasible items (weight <= capacity)
    # Note: In 0/1 knapsack, we can only pick items that fit.
    feasible_mask = weights <= remaining_capacity
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return -1
        
    # Hybrid Strategy:
    # If n is small, use exact DP to find the best item.
    # If n is large, use Fractional Lookahead.
    
    if n <= 25:
        # Use Exact DP
        sol = _dp_solve_exact(remaining_capacity, weights[feasible_indices], values[feasible_indices])
        if sol is None:
            # Fallback to greedy if DP failed (e.g. capacity too big)
            pass
        else:
            if len(sol) > 0:
                # Return the first item in the optimal solution
                # Note: sol contains indices relative to the feasible subset
                return int(feasible_indices[sol[0]])
            else:
                return -1
    else:
        # Use Fractional Lookahead for the top K candidates
        # Sort feasible items by density descending
        w_f = weights[feasible_indices]
        v_f = values[feasible_indices]
        
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = np.where(w_f > 0, v_f / w_f, np.inf)
            
        # Get top K candidates
        K = min(10, n)
        top_k_local = np.argsort(-densities)[:K]
        
        best_score = -np.inf
        best_idx = -1
        
        for local_idx in top_k_local:
            w_item = w_f[local_idx]
            v_item = v_f[local_idx]
            
            rem_cap = remaining_capacity - w_item
            
            if rem_cap < 0:
                continue
                
            # Create arrays for remaining items (excluding this one)
            # We can just pass the full feasible arrays and mask out the current one
            # But _fractional_knapsack_value sorts, so passing a filtered array is cleaner
            
            mask = np.ones(n, dtype=bool)
            mask[local_idx] = False
            w_rem = w_f[mask]
            v_rem = v_f[mask]
            
            bound = _fractional_knapsack_value(rem_cap, w_rem, v_rem)
            total_est = v_item + bound
            
            if total_est > best_score:
                best_score = total_est
                best_idx = local_idx
                
        if best_idx != -1:
            return int(feasible_indices[best_idx])
        else:
            # Fallback to best density
            best_density_idx = np.argmax(densities)
            return int(feasible_indices[best_density_idx])
