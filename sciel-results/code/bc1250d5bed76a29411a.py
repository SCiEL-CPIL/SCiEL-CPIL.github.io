import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    MOD = 1009
    N = MOD - 1  # 1008
    
    # Precompute modular inverses
    # inv[x] is the modular inverse of x mod 1009
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
        
    # Self-inverse elements are those where inv[x] == x
    # These are 1 and 1008
    is_self_inverse = [False] * (MOD + 1)
    for x in range(1, MOD):
        if inv[x] == x:
            is_self_inverse[x] = True

    best_start = 2
    best_len = 0
    
    # Precompute for each value v, whether it is "blocked" (self-inverse)
    # and its inverse
    
    # Two-pointer approach: for each start s, find max end e
    # such that [s, e] is valid.
    
    # A window [s, e] is valid if:
    # 1. No self-inverse element in [s, e]
    # 2. For no x in [s, e] is inv[x] also in [s, e]
    
    # Let's use a different approach:
    # For each pair (x, inv[x]) where x != inv[x], both cannot be in the interval.
    # This means the interval cannot contain both x and inv[x].
    
    # We can think of this as: we need to find the longest interval that avoids
    # containing both elements of any inverse pair, and avoids self-inverse elements.
    
    # Let's precompute the "conflict" pairs.
    # For each x in 1..1008, if x is not self-inverse, then (min(x,inv[x]), max(x,inv[x]))
    # is a conflict pair.
    
    # Actually, let's use the sliding window approach more carefully.
    # 
    # For a window [s, e], it's valid iff:
    # - For all x in [s,e]: not is_self_inverse[x]
    # - For all x in [s,e]: inv[x] not in [s,e] (or equivalently, inv[x] < s or inv[x] > e)
    
    # Let's use a count-based approach:
    # As we slide the window, we maintain a count of how many "conflicts" are active.
    
    # A conflict for element x is:
    # - If x is self-inverse: always a conflict
    # - Otherwise: a conflict if inv[x] is also in the window
    
    # Let's define for each position x:
    # - blocked[x] = True if x is self-inverse
    # - For the pair check: when we add x to the right, we check if inv[x] is in [s, e-1]
    #   When we remove x from the left, we check if inv[x] is in [s+1, e]
    
    # Actually, let's just do the direct O(N^2) but with early termination and
    # vectorized checks using NumPy for speed.
    
    import numpy as np
    
    # Create array of valid individual elements
    valid_elem = np.array([not is_self_inverse[i] for i in range(1, MOD)], dtype=bool)
    
    # For each start s (0-indexed in our array, representing value s+1),
    # find the maximum end e such that [s, e] is valid.
    
    # We can precompute for each value v, its inverse index (0-based)
    # inv_idx[v] = inv[v+1] - 1 (converting to 0-based index)
    inv_idx = np.array([inv[i+1] - 1 for i in range(N)], dtype=np.int32)
    
    best_len = 0
    best_start = 2  # 1-based
    
    # For each start position s (0-based)
    for s in range(N):
        if time.time() > deadline:
            break
            
        if not valid_elem[s]:
            continue
            
        # Find max e >= s such that [s, e] is valid
        # We need: for all x in [s, e]:
        #   - valid_elem[x] is True
        #   - inv_idx[x] < s or inv_idx[x] > e
        
        # Start with e = s, extend as far as possible
        e = s
        
        # Check if we can extend
        while e + 1 < N:
            cand = e + 1
            
            if not valid_elem[cand]:
                break
                
            # Check if inv[cand] is in [s, e]
            inv_cand = inv_idx[cand]
            if s <= inv_cand <= e:
                break
                
            e = cand
            
        if e >= s:
            length = e - s + 1
            if length > best_len:
                best_len = length
                best_start = s + 1  # Convert to 1-based
                
    # Fallback
    if best_len == 0:
        best_start = 2
        best_len = 1
        
    return [best_start, best_len]
