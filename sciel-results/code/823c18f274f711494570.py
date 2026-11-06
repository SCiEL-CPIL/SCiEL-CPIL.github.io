import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    n = 512
    k = 4
    
    # Represent the set S as a Python integer bitmask.
    # Bit i is 1 if i is in S.
    # This allows very fast bitwise operations.
    
    # Precompute for each element x, the list of APs it participates in.
    # An AP is (a, a+d, a+2d, a+3d).
    # If we add x, it creates a conflict if there exists an AP containing x
    # where the other 3 elements are already in S.
    
    # To check if adding x is valid:
    # We need to check all APs containing x.
    # For an AP (a, b, c, d) containing x, if the other 3 are in S, it's invalid.
    # 
    # Let's precompute for each x, the masks of the "other 3" for each AP containing x.
    # Actually, simpler:
    # For each x, store a list of tuples (mask1, mask2, mask3) where mask_i is a bitmask
    # representing the other 3 elements of a specific AP containing x.
    # If (S & mask_i) == mask_i, then adding x creates a 4-AP.
    
    # Even better: Precompute a list of "conflict masks" for each x.
    # conflict_masks[x] is a list of bitmasks M such that if (S & M) == M, x cannot be added.
    
    conflict_masks = [[] for _ in range(n + 1)]
    
    # Generate all 4-term APs
    # d from 1 to 170
    # a from 1 to n - 3d
    for d in range(1, (n - 1) // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            
            # AP is (a, b, c, e)
            # For element a: other 3 are b, c, e
            mask_a = (1 << b) | (1 << c) | (1 << e)
            conflict_masks[a].append(mask_a)
            
            # For element b: other 3 are a, c, e
            mask_b = (1 << a) | (1 << c) | (1 << e)
            conflict_masks[b].append(mask_b)
            
            # For element c: other 3 are a, b, e
            mask_c = (1 << a) | (1 << b) | (1 << e)
            conflict_masks[c].append(mask_c)
            
            # For element e: other 3 are a, b, c
            mask_e = (1 << a) | (1 << b) | (1 << c)
            conflict_masks[e].append(mask_e)
    
    def is_valid_add(x, S_mask):
        """Check if adding x to S_mask creates a 4-AP."""
        for m in conflict_masks[x]:
            if (S_mask & m) == m:
                return False
        return True
        
    def popcount(x):
        return bin(x).count('1')
        
    def base3_init():
        """
        Construct a 4-AP-free set using base-3 digits.
        Select numbers in [1, n] that do not have digit '2' in base 3.
        This is a variant of the Cantor set.
        """
        S_mask = 0
        for i in range(1, n + 1):
            # Check if i has no '2' in base 3
            valid = True
            temp = i
            while temp > 0:
                if temp % 3 == 2:
                    valid = False
                    break
                temp //= 3
            if valid:
                S_mask |= (1 << i)
        return S_mask

    def greedy_init():
        S_mask = 0
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        for x in candidates:
            if is_valid_add(x, S_mask):
                S_mask |= (1 << x)
        return S_mask

    # Start with base-3 construction as it provides a strong structured lower bound
    best_mask = base3_init()
    best_size = popcount(best_mask)
    
    # Also try greedy to see if it's better, but base-3 is usually better for AP-free
    g_mask = greedy_init()
    g_size = popcount(g_mask)
    if g_size > best_size:
        best_mask = g_mask
        best_size = g_size

    cur_mask = best_mask
    cur_size = best_size
    
    # Precompute list of all elements for random selection
    all_elements = list(range(1, n + 1))
    
    # To make it faster, we can use NumPy for the conflict check?
    # No, Python int bitwise is very fast.
    
    # Let's refine the search strategy:
    # 1. Try to add a random unselected element.
    # 2. If no additions possible, try swaps:
    #    - Remove y, add x1, x2 (size +1)
    #    - Remove y, add x1 (size 0, neutral move for exploration)
    
    # We need to efficiently find candidates.
    # Unselected: bits in ~S_mask & ((1 << (n+1)) - 1) - 1 (mask out bit 0)
    # Selected: bits in S_mask
    
    def get_unselected(S_mask):
        # Return list of unselected elements
        # This is O(n).
        return [i for i in range(1, n + 1) if not (S_mask & (1 << i))]
        
    def get_selected(S_mask):
        return [i for i in range(1, n + 1) if (S_mask & (1 << i))]
    
    # To avoid O(n) list generation every time, we can just iterate 1..n and check bit.
    
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    while time.time() < deadline - 2:
        improved = False
        
        # Phase 1: Try to add
        # Iterate through unselected elements in random order
        unselected = [i for i in range(1, n + 1) if not (cur_mask & (1 << i))]
        random.shuffle(unselected)
        
        for x in unselected:
            if is_valid_add(x, cur_mask):
                cur_mask |= (1 << x)
                cur_size += 1
                if cur_size > best_size:
                    best_size = cur_size
                    best_mask = cur_mask
                improved = True
                break
        
        if improved:
            continue
            
        # Phase 2: Try 1-for-2 swap (Increase size by 1)
        # Remove y, add x1, x2
        selected = [i for i in range(1, n + 1) if (cur_mask & (1 << i))]
        random.shuffle(selected)
        
        # We need to find y such that removing it allows adding 2 new elements.
        # This is harder. 
        # Strategy: Pick random y. Remove it. Try to add 2 random elements.
        # If valid, keep.
        
        found_swap = False
        for y in selected[:20]: # Limit attempts
            if time.time() > deadline - 2:
                break
                
            # Remove y
            temp_mask = cur_mask & ~(1 << y)
            
            # Try to add 2 elements from unselected
            # We can just pick 2 random unselected and check.
            # To be more systematic, we could check all pairs, but that's O(n^2).
            # Random sampling is better.
            
            unselected_after_remove = [i for i in range(1, n + 1) if not (temp_mask & (1 << i))]
            if len(unselected_after_remove) < 2:
                continue
            
            # Try a few random pairs
            for _ in range(10):
                x1, x2 = random.sample(unselected_after_remove, 2)
                
                # Check if adding x1 is valid to temp_mask
                if not is_valid_add(x1, temp_mask):
                    continue
                
                temp_mask1 = temp_mask | (1 << x1)
                
                # Check if adding x2 is valid to temp_mask1
                if not is_valid_add(x2, temp_mask1):
                    continue
                
                # Valid swap found!
                cur_mask = temp_mask1 | (1 << x2)
                cur_size += 1
                if cur_size > best_size:
                    best_size = cur_size
                    best_mask = cur_mask
                improved = True
                found_swap = True
                break
            
            if found_swap:
                break
        
        if improved:
            continue
            
        # Phase 3: Neutral moves (1-for-1 swap) to escape local optima
        # Accept with probability exp(-delta/T) ? No, delta is 0.
        # Just accept if valid.
        
        if selected:
            y = random.choice(selected)
            temp_mask = cur_mask & ~(1 << y)
            
            unselected_after_remove = [i for i in range(1, n + 1) if not (temp_mask & (1 << i))]
            if unselected_after_remove:
                x = random.choice(unselected_after_remove)
                if is_valid_add(x, temp_mask):
                    # Accept neutral move
                    cur_mask = temp_mask | (1 << x)
                    # Size unchanged
                    # Don't update best_mask unless we want to track path? 
                    # No, best_mask should be the max size found.
                    # But we can continue from this new state.
                    pass
        
        # If stuck for too long, kick
        # Remove 3 random elements to open up space
        if selected:
            kicks = random.sample(selected, min(3, len(selected)))
            for k in kicks:
                cur_mask &= ~(1 << k)
            cur_size -= len(kicks)
            
        T *= cooling
        if T < min_T:
            T = min_T
            
    # Final verification of best_mask
    # best_mask is guaranteed valid by construction (we only accepted valid moves)
    # But just in case, verify.
    
    # Convert best_mask to list
    result = [i for i in range(1, n + 1) if (best_mask & (1 << i))]
    
    # Double check validity
    S_set = set(result)
    valid = True
    for a in S_set:
        for d in range(1, (n - a) // 3 + 1):
            if (a+d) in S_set and (a+2*d) in S_set and (a+3*d) in S_set:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback: greedy
        S_mask = 0
        for x in range(1, n+1):
            if is_valid_add(x, S_mask):
                S_mask |= (1 << x)
        result = [i for i in range(1, n + 1) if (S_mask & (1 << i))]
        
    return sorted(result)

# Test run
if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    # Verify
    S = set(res)
    valid = True
    for a in S:
        for d in range(1, (512 - a) // 3 + 1):
            if (a+d) in S and (a+2*d) in S and (a+3*d) in S:
                valid = False
                break
        if not valid:
            break
    print(f"Valid: {valid}")
