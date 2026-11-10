import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s limit
    
    # 255 = 3 * 5 * 17
    # Every element in the set must share a factor with 255, i.e., be divisible by 3, 5, or 17.
    # We need pairwise gcd > 1 for all distinct pairs, and 255 must be in the set.
    
    # Generate candidates: numbers in [1, 255] that share a factor with 255
    candidates = [x for x in range(1, 256) if math.gcd(x, 255) > 1]
    n = len(candidates)
    
    # Map values to indices
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    idx_255 = val_to_idx[255]
    
    # Precompute adjacency using bitmasks for fast compatibility checks
    # adj_mask[i] is a bitmask where bit j is set if gcd(candidates[i], candidates[j]) > 1
    adj_mask = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if i != j and math.gcd(candidates[i], candidates[j]) > 1:
                mask |= (1 << j)
        adj_mask[i] = mask
    
    # Helper: check if vertex i is compatible with all vertices in a set represented as bitmask
    def is_compatible(i, set_mask):
        # i must be adjacent to all vertices in set_mask
        # This is true if set_mask is a subset of adj_mask[i]
        return (set_mask & ~adj_mask[i]) == 0
    
    # Helper: get the set of vertices from a bitmask
    def mask_to_list(mask):
        result = []
        for i in range(n):
            if mask & (1 << i):
                result.append(candidates[i])
        return result
    
    # Helper: convert a list of values to a bitmask
    def list_to_mask(val_list):
        mask = 0
        for v in val_list:
            mask |= (1 << val_to_idx[v])
        return mask
    
    # Baseline: all multiples of 3 (size 85)
    baseline = [x for x in range(1, 256) if x % 3 == 0]
    best_mask = list_to_mask(baseline)
    best_size = len(baseline)
    
    # Try to improve by adding vertices to the baseline
    # A vertex can be added if it's compatible with all current vertices
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        for i in range(n):
            if best_mask & (1 << i):
                continue
            if is_compatible(i, best_mask):
                best_mask |= (1 << i)
                best_size += 1
                improved = True
    
    # Now try local search: for each vertex in the set, try removing it and see if we can add more
    # This is a 1-swap move: remove one, try to add multiple
    for _ in range(100):
        if time.time() > deadline - 10.0:
            break
        
        current_mask = best_mask
        current_size = best_size
        improved = False
        
        # Try removing each vertex and then greedily adding compatible ones
        for i in range(n):
            if not (current_mask & (1 << i)):
                continue
            
            # Remove vertex i
            trial_mask = current_mask & ~(1 << i)
            trial_size = current_size - 1
            
            # Greedily add compatible vertices
            for j in range(n):
                if trial_mask & (1 << j):
                    continue
                if is_compatible(j, trial_mask):
                    trial_mask |= (1 << j)
                    trial_size += 1
            
            if trial_size > current_size:
                current_mask = trial_mask
                current_size = trial_size
                improved = True
                break
        
        if current_size > best_size:
            best_mask = current_mask
            best_size = current_size
            improved = True
        
        if not improved:
            break
    
    # Try multiple random restarts
    random.seed(42)
    for restart in range(200):
        if time.time() > deadline - 5.0:
            break
        
        # Start with 255 and randomly add compatible vertices
        current_mask = 1 << idx_255
        current_size = 1
        
        # Get list of other vertices in random order
        others = [i for i in range(n) if i != idx_255]
        random.shuffle(others)
        
        for i in others:
            if time.time() > deadline - 5.0:
                break
            if is_compatible(i, current_mask):
                current_mask |= (1 << i)
                current_size += 1
        
        # Local search: try to add more vertices
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(n):
                if current_mask & (1 << i):
                    continue
                if is_compatible(i, current_mask):
                    current_mask |= (1 << i)
                    current_size += 1
                    improved = True
        
        if current_size > best_size:
            best_mask = current_mask
            best_size = current_size
    
    # Convert best_mask to list
    result = mask_to_list(best_mask)
    
    # Ensure 255 is in the result
    if 255 not in result:
        result.append(255)
    
    return result
