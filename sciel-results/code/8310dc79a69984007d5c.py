import time
import sys

sys.setrecursionlimit(10000)

def construct():
    deadline = time.time() + 85
    n = 256
    
    # 1. Precompute conflict masks
    conflict_mask = [0] * (n + 1)
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                conflict_mask[a] |= (1 << b)
                conflict_mask[b] |= (1 << a)
    
    full_mask = 0
    for i in range(1, n + 1):
        full_mask |= (1 << i)
        
    best_size = 0
    best_set_mask = 0
    
    # Helper to convert mask to list
    def mask_to_list(mask):
        return [i for i in range(1, n + 1) if (mask >> i) & 1]
    
    # Initial greedy solution to set a strong lower bound
    # Sort by degree ascending
    degrees = [(conflict_mask[i].bit_count(), i) for i in range(1, n + 1)]
    degrees.sort()
    greedy_set = 0
    for deg, i in degrees:
        if (greedy_set & conflict_mask[i]) == 0:
            greedy_set |= (1 << i)
            
    best_size = greedy_set.bit_count()
    best_set_mask = greedy_set
    
    # B&B Search
    def solve(remaining_mask, current_mask, current_size):
        nonlocal best_size, best_set_mask, deadline
        
        if time.time() > deadline:
            return
            
        # Pruning
        rem_count = remaining_mask.bit_count()
        if current_size + rem_count <= best_size:
            return
            
        if remaining_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set_mask = current_mask
            return
            
        # Heuristic: Pick vertex with highest degree in remaining_mask
        max_deg = -1
        v = -1
        m = remaining_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            deg = (conflict_mask[idx] & remaining_mask).bit_count()
            if deg > max_deg:
                max_deg = deg
                v = idx
            m ^= lsb
            
        if v == -1:
            return
            
        lsb = 1 << v
        
        # Branch 1: Include v
        # Check if compatible (should be, as v is in remaining_mask)
        new_mask_include = remaining_mask & ~conflict_mask[v] & ~lsb
        new_current_mask_include = current_mask | lsb
        solve(new_mask_include, new_current_mask_include, current_size + 1)
        
        # Branch 2: Exclude v
        new_mask_exclude = remaining_mask & ~lsb
        solve(new_mask_exclude, current_mask, current_size)
        
    solve(full_mask, 0, 0)
    
    return mask_to_list(best_set_mask)
