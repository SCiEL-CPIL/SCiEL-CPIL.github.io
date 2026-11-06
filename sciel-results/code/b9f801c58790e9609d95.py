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
    
    # Precompute conflict masks for each element.
    # conflict_masks[x] is a list of bitmasks M such that if (S & M) == M, x cannot be added.
    conflict_masks = [[] for _ in range(n + 1)]
    
    for d in range(1, (n - 1) // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            
            mask_a = (1 << b) | (1 << c) | (1 << e)
            conflict_masks[a].append(mask_a)
            
            mask_b = (1 << a) | (1 << c) | (1 << e)
            conflict_masks[b].append(mask_b)
            
            mask_c = (1 << a) | (1 << b) | (1 << e)
            conflict_masks[c].append(mask_c)
            
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
        """
        S_mask = 0
        for i in range(1, n + 1):
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

    def greedy_priority_init():
        """
        Greedy with priority: try to add elements that are 'central' or have fewer conflicts?
        Actually, a better heuristic for AP-free sets is to prioritize elements that are 
        'sparse' in terms of potential APs, or simply use a random order but with a 
        'lookahead' or 'backtracking' lite.
        
        Let's try a different greedy: iterate 1..n, but skip if it creates an AP.
        This is deterministic and fast.
        """
        S_mask = 0
        for x in range(1, n + 1):
            if is_valid_add(x, S_mask):
                S_mask |= (1 << x)
        return S_mask

    # Start with base-3 construction
    best_mask = base3_init()
    best_size = popcount(best_mask)
    
    # Also try greedy to see if it's better
    g_mask = greedy_init()
    g_size = popcount(g_mask)
    if g_size > best_size:
        best_mask = g_mask
        best_size = g_size
        
    g2_mask = greedy_priority_init()
    g2_size = popcount(g2_mask)
    if g2_size > best_size:
        best_mask = g2_mask
        best_size = g2_size

    cur_mask = best_mask
    cur_size = best_size
    
    # Local search refinement
    # We will use a simulated annealing style approach with swaps.
    
    def get_unselected(S_mask):
        return [i for i in range(1, n + 1) if not (S_mask & (1 << i))]
        
    def get_selected(S_mask):
        return [i for i in range(1, n + 1) if (S_mask & (1 << i))]
    
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    while time.time() < deadline - 2:
        improved = False
        
        # Phase 1: Try to add
        unselected = get_unselected(cur_mask)
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
        selected = get_selected(cur_mask)
        random.shuffle(selected)
        
        found_swap = False
        for y in selected[:30]: # Limit attempts
            if time.time() > deadline - 2:
                break
                
            temp_mask = cur_mask & ~(1 << y)
            
            unselected_after_remove = get_unselected(temp_mask)
            if len(unselected_after_remove) < 2:
                continue
            
            # Try a few random pairs
            for _ in range(15):
                x1, x2 = random.sample(unselected_after_remove, 2)
                
                if not is_valid_add(x1, temp_mask):
                    continue
                
                temp_mask1 = temp_mask | (1 << x1)
                
                if not is_valid_add(x2, temp_mask1):
                    continue
                
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
        if selected:
            y = random.choice(selected)
            temp_mask = cur_mask & ~(1 << y)
            
            unselected_after_remove = get_unselected(temp_mask)
            if unselected_after_remove:
                x = random.choice(unselected_after_remove)
                if is_valid_add(x, temp_mask):
                    cur_mask = temp_mask | (1 << x)
                    # Size unchanged
                    pass
        
        # If stuck for too long, kick
        if selected:
            kicks = random.sample(selected, min(3, len(selected)))
            for k in kicks:
                cur_mask &= ~(1 << k)
            cur_size -= len(kicks)
            
        T *= cooling
        if T < min_T:
            T = min_T
            
    # Final verification of best_mask
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
