import time
import math
import random

_START = time.time()
_DEADLINE = 85.0

def _time_left():
    return _DEADLINE - (time.time() - _START)

N = 24
_L = 1
for k in range(1, 25):
    _L = _L * k // math.gcd(_L, k)
_W = [_L // k for k in range(1, 25)]

def _subset_sums(elements):
    """Returns the set of all subset sums of elements."""
    sums = {0}
    for x in elements:
        # Optimization: if x is 0, skip? No, weights are > 0.
        new_sums = set()
        for s in sums:
            new_sums.add(s + x)
        sums.update(new_sums)
    return sums

def _is_valid_addition(k_idx, selected_indices):
    """
    Check if adding k_idx to selected_indices is valid.
    selected_indices does not contain k_idx.
    """
    S = selected_indices
    w_k = _W[k_idx]
    
    # 1. Check if w_k is a subset sum of S
    if S:
        sums_S = _subset_sums([_W[i] for i in S])
        if w_k in sums_S:
            return False
    else:
        sums_S = {0} # Not strictly needed for this check if S is empty
        
    # 2. Check if any y in S becomes subsumed
    for y in S:
        w_y = _W[y]
        # Check if w_y is a subset sum of (S \ {y}) U {k_idx}
        # This is true if w_y in sums_{S\{y}} (False) OR w_y - w_k in sums_{S\{y}}
        diff = w_y - w_k
        if diff < 0:
            continue
        # Need to check if diff is in sums_{S \ {y}}
        others = [i for i in S if i != y]
        if not others:
            if diff == 0:
                return False
            else:
                continue
        sums_others = _subset_sums([_W[i] for i in others])
        if diff in sums_others:
            return False
            
    return True

def construct():
    best_set = []
    
    # Try multiple orders
    orders = [
        list(range(23, -1, -1)), # Descending index (24 to 1)
        list(range(24)),         # Ascending index (1 to 24)
    ]
    
    # Add some random orders
    random.seed(42)
    for _ in range(20):
        order = list(range(24))
        random.shuffle(order)
        orders.append(order)
        
    for order in orders:
        if _time_left() < 5.0:
            break
            
        current = []
        for k in order:
            if _time_left() < 2.0:
                break
            if _is_valid_addition(k, current):
                current.append(k)
                
        if len(current) > len(best_set):
            best_set = current
            
        # Local search: try to add any remaining element
        changed = True
        while changed and _time_left() > 2.0:
            changed = False
            for k in range(24):
                if k in current:
                    continue
                if _is_valid_addition(k, current):
                    current.append(k)
                    changed = True
                    if len(current) > len(best_set):
                        best_set = current
                    break # Restart scan as set changed
                    
    if not best_set:
        return [24]
        
    return [i + 1 for i in best_set]
