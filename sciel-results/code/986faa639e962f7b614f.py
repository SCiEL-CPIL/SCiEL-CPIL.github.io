# EVOLVE-BLOCK-START
import numpy as np
import time

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Finds a maximum square-difference-free set modulo 205 using a 
    Bitset-based Maximum Independent Set solver with branch and bound.
    """
    m = 205
    
    # 1. Compute Forbidden Differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # 2. Build Conflict Masks
    conflict_masks = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i == j:
                continue
            d = (i - j) % m
            if d in forbidden:
                mask |= (1 << j)
        conflict_masks[i] = mask
        
    # 3. Strong Greedy Initial Solution
    import random
    best_greedy = []
    # Run many random greedy tries
    for _ in range(500):
        remaining = list(range(m))
        random.shuffle(remaining)
        curr = []
        curr_set = set()
        # To speed up, we can check conflicts against the set
        for v in remaining:
            # Check if v conflicts with any in curr
            # Using bitmasks is faster
            # But curr is small, so loop is fine
            ok = True
            for u in curr:
                if (conflict_masks[v] >> u) & 1:
                    ok = False
                    break
            if ok:
                curr.append(v)
                curr_set.add(v)
        if len(curr) > len(best_greedy):
            best_greedy = curr
            # Optional: early stop if we hit a known high bound?
            
    best_size = len(best_greedy)
    best_set = set(best_greedy)
    
    # 4. Branch and Bound Search
    start_time = time.time()
    time_limit = 500.0  # Leave 100s buffer
    
    current_path = []
    
    def search(available_mask, current_size):
        nonlocal best_size, best_set
        nonlocal time_limit, start_time
        
        # Time check
        if time.time() - start_time > time_limit:
            return
            
        # Pruning: Upper bound is current_size + popcount(available)
        avail_count = available_mask.bit_count()
        if current_size + avail_count <= best_size:
            return
            
        if available_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = set(current_path)
            return
            
        # Select vertex with max conflicts in available_mask
        best_v = -1
        max_conflicts = -1
        temp_mask = available_mask
        while temp_mask:
            lsb = temp_mask & -temp_mask
            v = lsb.bit_length() - 1
            conflicts = (conflict_masks[v] & available_mask).bit_count()
            if conflicts > max_conflicts:
                max_conflicts = conflicts
                best_v = v
            temp_mask ^= lsb
            
        if best_v == -1:
            return

        # Branch 1: Include best_v
        new_mask_include = available_mask & ~(conflict_masks[best_v] | (1 << best_v))
        current_path.append(best_v)
        search(new_mask_include, current_size + 1)
        current_path.pop()
        
        # Branch 2: Exclude best_v
        new_mask_exclude = available_mask & ~(1 << best_v)
        search(new_mask_exclude, current_size)

    # Start search
    all_mask = (1 << m) - 1
    search(all_mask, 0)
    
    # If time ran out, we return the best found so far
    result = np.array(sorted(best_set), dtype=np.int64)
    
    # Final Validation
    final_set = set(result)
    for a in final_set:
        for b in final_set:
            if a >= b: continue
            d = (a - b) % m
            if d in forbidden:
                # Should not happen
                return np.array(sorted(best_greedy), dtype=np.int64)
                
    return result

# EVOLVE-BLOCK-END
