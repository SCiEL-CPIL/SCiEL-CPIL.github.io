import numpy as np
import time

_START_TIME = None
_TIME_LIMIT = 100.0 # Safety margin

def _get_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time()

def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    # 1. Basic checks
    if remaining_capacity <= 0:
        return -1
    
    n = len(remaining_weights)
    if n == 0:
        return -1

    # 2. Filter feasible items
    # Convert to lists for easier manipulation if needed, or use numpy masks
    weights = np.array(remaining_weights)
    values = np.array(remaining_values)
    
    # Mask for items that fit
    mask = weights <= remaining_capacity
    if not np.any(mask):
        return -1
    
    feasible_indices = np.where(mask)[0]
    feasible_weights = weights[mask]
    feasible_values = values[mask]
    
    # 3. Check for zero-weight items (after scaling)
    SCALE = 100
    cap_int = int(remaining_capacity * SCALE)
    
    # Calculate scaled weights
    # Use floor to ensure we don't exceed capacity due to rounding up
    scaled_weights = np.floor(feasible_weights * SCALE).astype(int)
    
    # Identify zero-weight items in scaled space
    zero_mask = scaled_weights == 0
    if np.any(zero_mask):
        # Return the first zero-weight item (highest value? or just first?)
        # Since they are free, we should take the one with highest value?
        # But the interface asks for "next item".
        # If we return one, the caller removes it.
        # We should return the one that is part of an optimal solution.
        # All zero-weight items are part of an optimal solution (if value > 0).
        # Let's return the index of the zero-weight item with the highest value.
        zero_indices_in_feasible = np.where(zero_mask)[0]
        if len(zero_indices_in_feasible) > 0:
            # Get values of these zero-weight items
            zv = feasible_values[zero_indices_in_feasible]
            best_zero_local = zero_indices_in_feasible[np.argmax(zv)]
            # Map back to original index
            return int(feasible_indices[best_zero_local])

    # 4. DP for positive weight items
    # Filter out zero-weight items from DP consideration
    pos_mask = scaled_weights > 0
    if not np.any(pos_mask):
        # All feasible items had 0 scaled weight, handled above
        return -1
        
    dp_weights = scaled_weights[pos_mask]
    dp_values = feasible_values[pos_mask]
    dp_indices = feasible_indices[pos_mask]
    
    N = len(dp_weights)
    C = cap_int
    
    # If C is 0, no positive weight items can fit
    if C == 0:
        return -1

    # DP Table
    # dp[c] = max value
    # take[i][c] = boolean, was item i taken for capacity c?
    # To save memory, we can store backpointers, but boolean matrix is fine for N=100, C=1250
    # 100 * 1250 = 125k bools.
    
    # Initialize
    dp = np.zeros(C + 1, dtype=float)
    take = np.zeros((N, C + 1), dtype=bool)
    
    # Fill DP
    for i in range(N):
        w = dp_weights[i]
        v = dp_values[i]
        # Iterate backwards
        for c in range(C, w - 1, -1):
            if dp[c - w] + v > dp[c]:
                dp[c] = dp[c - w] + v
                take[i][c] = True
            # else take[i][c] remains False (inherit from previous item implicitly)
            # Wait, standard 1D DP overwrites dp[c].
            # To backtrack, we need to know if item i was used.
            # If dp[c] was updated using item i, take[i][c] = True.
            # If not, take[i][c] = False.
            # This works because if we don't update, the value comes from dp_old[c], 
            # which corresponds to "not taking item i".
            # So take[i][c] = False is correct.

    # Backtrack
    c = C
    # We want to find ANY item in the optimal set.
    # We traverse i from N-1 down to 0.
    for i in range(N - 1, -1, -1):
        if take[i][c]:
            # Item i is in the optimal solution
            return int(dp_indices[i])
        else:
            # Item i was not taken, capacity remains c
            # c = c (no change)
            pass
        # If take[i][c] is True, we subtract weight?
        # No, we just need to return ONE index.
        # We found one. Return it.
        
    # If no item found (should not happen if dp[C] > 0)
    # Fallback to greedy
    densities = dp_values / dp_weights
    best_idx = np.argmax(densities)
    return int(dp_indices[best_idx])
