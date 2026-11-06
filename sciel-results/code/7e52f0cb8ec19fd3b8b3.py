"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = None
_TIME_LIMIT = 115.0  # safety margin before 120s limit


def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()


def _check_time():
    """Return True if we should continue (time remaining > 1s margin)."""
    if _START_TIME is None:
        return True
    return (time.time() - _START_TIME) < _TIME_LIMIT


def _simulate_greedy_value(capacity, weights, values, excluded_idx):
    """
    Simulate the greedy density algorithm on the given arrays,
    excluding the item at excluded_idx.
    Returns the total value accumulated.
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0
    
    # Create a mask to exclude the item
    mask = np.ones(n, dtype=bool)
    if excluded_idx is not None and 0 <= excluded_idx < n:
        mask[excluded_idx] = False
    
    # Get indices of non-excluded items
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return 0.0
    
    w = weights[idx]
    v = values[idx]
    
    # Filter out items that are too heavy
    valid = w <= capacity
    if not np.any(valid):
        return 0.0
    
    w = w[valid]
    v = v[valid]
    
    # Compute density
    density = v / w
    
    # Sort by density descending
    order = np.argsort(-density)
    
    total_value = 0.0
    remaining_cap = capacity
    
    for i in order:
        if w[i] <= remaining_cap:
            remaining_cap -= w[i]
            total_value += v[i]
        else:
            # In 0/1 knapsack, we can't take partial items
            # But for greedy simulation, we stop when we can't fit
            # Actually, we should continue checking other items
            # Let me reconsider: in standard greedy for 0/1, we just skip if doesn't fit
            pass
    
    return total_value


def _simulate_greedy_value_optimized(capacity, weights, values, excluded_idx):
    """
    More optimized simulation using vectorized operations where possible.
    """
    n = len(weights)
    if n == 0 or capacity <= 0:
        return 0.0
    
    # Create a mask to exclude the item
    if excluded_idx is not None and 0 <= excluded_idx < n:
        mask = np.ones(n, dtype=bool)
        mask[excluded_idx] = False
    else:
        mask = np.ones(n, dtype=bool)
    
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return 0.0
    
    w = weights[idx]
    v = values[idx]
    
    # Filter out items that are too heavy
    valid = w <= capacity
    if not np.any(valid):
        return 0.0
    
    w = w[valid]
    v = v[valid]
    
    if len(w) == 0:
        return 0.0
    
    # Compute density
    density = v / w
    
    # Sort by density descending
    order = np.argsort(-density)
    
    total_value = 0.0
    remaining_cap = capacity
    
    # Use a loop but with early termination
    for i in order:
        wi = w[i]
        if wi <= remaining_cap:
            remaining_cap -= wi
            total_value += v[i]
            if remaining_cap <= 0:
                break
    
    return total_value


def select_next_item(remaining_capacity, remaining_weights, remaining_values):
    """
    Pick the next item to add to the knapsack, or -1 to stop.
    
    Strategy: For each candidate item, simulate what the final greedy solution
    would look like if we picked that item first, then continue greedily.
    Pick the item that leads to the best total value.
    """
    _init_time()
    
    if remaining_capacity <= 0:
        return -1
    
    n = len(remaining_weights)
    if n == 0:
        return -1
    
    # If only one item, pick it if it fits
    if n == 1:
        if remaining_weights[0] <= remaining_capacity:
            return 0
        return -1
    
    # Check if any item fits
    fits = remaining_weights <= remaining_capacity
    if not np.any(fits):
        return -1
    
    # Get indices of items that fit
    fitting_idx = np.where(fits)[0]
    
    if len(fitting_idx) == 0:
        return -1
    
    # If only one item fits, pick it
    if len(fitting_idx) == 1:
        return int(fitting_idx[0])
    
    # For small n, evaluate each candidate
    # For larger n, use a heuristic to limit candidates
    
    # First, compute densities for all items
    densities = remaining_values / remaining_weights
    
    # Sort fitting items by density descending
    fitting_densities = densities[fitting_idx]
    sorted_order = np.argsort(-fitting_densities)
    
    # Limit the number of candidates to evaluate based on time and n
    # For n <= 50, evaluate all fitting items
    # For larger n, evaluate top K by density
    if n <= 50:
        candidates = fitting_idx[sorted_order]
    else:
        # Evaluate top 20 by density plus some random samples
        top_k = min(20, len(fitting_idx))
        candidates = fitting_idx[sorted_order[:top_k]]
    
    best_value = -1.0
    best_idx = -1
    
    for idx in candidates:
        if not _check_time():
            break
        
        # Simulate: pick this item first, then continue greedily
        # After picking this item, remaining capacity is reduced
        new_cap = remaining_capacity - remaining_weights[idx]
        
        if new_cap < 0:
            continue
        
        # Simulate greedy on remaining items (excluding idx)
        # We need to simulate what the greedy algorithm would do
        # starting from the state where we've already picked idx
        
        # Create modified arrays excluding idx
        mask = np.ones(n, dtype=bool)
        mask[idx] = False
        idxs = np.where(mask)[0]
        
        if len(idxs) == 0:
            # Only this item, so total value is just this item's value
            total_val = remaining_values[idx]
        else:
            w_sub = remaining_weights[idxs]
            v_sub = remaining_values[idxs]
            
            # Filter to items that fit in new capacity
            valid = w_sub <= new_cap
            if not np.any(valid):
                total_val = remaining_values[idx]
            else:
                w_sub = w_sub[valid]
                v_sub = v_sub[valid]
                
                if len(w_sub) == 0:
                    total_val = remaining_values[idx]
                else:
                    # Compute density and sort
                    dens_sub = v_sub / w_sub
                    order_sub = np.argsort(-dens_sub)
                    
                    total_val = remaining_values[idx]
                    rem_cap = new_cap
                    
                    for j in order_sub:
                        if w_sub[j] <= rem_cap:
                            rem_cap -= w_sub[j]
                            total_val += v_sub[j]
                            if rem_cap <= 0:
                                break
        
        if total_val > best_value:
            best_value = total_val
            best_idx = int(idx)
    
    if best_idx == -1:
        # Fallback: pick highest density item that fits
        return int(np.argmax(densities[fitting_idx]))
    
    return best_idx
# EVOLVE-BLOCK-END
