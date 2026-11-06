import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    # Bitset of reachable sums. Bit i is 1 if sum i is reachable.
    MASK = (1 << 201) - 1  # Bits 0 to 200 inclusive
    TARGET_BIT = 1 << 200
    
    def compute_reachable(selected):
        reachable = 1
        for x in selected:
            reachable |= (reachable << x)
            reachable &= MASK
        return reachable

    def is_valid_addition(reachable, x):
        new_reachable = reachable | (reachable << x)
        new_reachable &= MASK
        return not (new_reachable & TARGET_BIT)

    # 1. Heuristic Initialization: 
    # We hypothesize that {66, ..., 100} is a strong candidate.
    # Let's start with this set.
    selected = list(range(66, 101))
    reachable = compute_reachable(selected)
    
    # Verify validity
    if reachable & TARGET_BIT:
        # Fallback to greedy if this specific set is invalid (it shouldn't be)
        selected = []
        reachable = 1
        for x in range(100, 0, -1):
            if is_valid_addition(reachable, x):
                selected.append(x)
                reachable |= (reachable << x)
                reachable &= MASK
    
    best_selected = selected[:]
    best_size = len(selected)
    best_reachable = reachable
    
    # 2. Local Search to try and improve (find size > 35)
    # We try to add elements from [1, 65] if valid.
    # We try swaps: remove one from selected, add one or two from excluded.
    
    random.seed(42)
    excluded = [x for x in range(1, 101) if x not in set(selected)]
    
    # Phase 2a: Try to add any excluded element
    changed = True
    while changed and (time.time() - start_time) < deadline:
        changed = False
        for x in excluded[:]:
            if is_valid_addition(best_reachable, x):
                best_selected.append(x)
                best_reachable |= (best_reachable << x)
                best_reachable &= MASK
                best_size += 1
                excluded.remove(x)
                changed = True
                
    # Phase 2b: Randomized Swaps (Remove 1, Add 2)
    # This is the most likely way to increase size if a better configuration exists.
    iterations = 0
    max_iterations = 20000
    
    while iterations < max_iterations and (time.time() - start_time) < deadline:
        iterations += 1
        
        if not best_selected:
            break
            
        # Pick a random element to remove
        rem_idx = random.randint(0, len(best_selected) - 1)
        rem_val = best_selected[rem_idx]
        
        # Compute reachable without rem_val
        # Optimization: We could use prefix/suffix sums, but N is small.
        # We recompute for simplicity and correctness.
        temp_selected = best_selected[:]
        temp_selected.pop(rem_idx)
        
        temp_reachable = 1
        for x in temp_selected:
            temp_reachable |= (temp_reachable << x)
            temp_reachable &= MASK
            
        # Try to add 2 random excluded elements
        if len(excluded) < 2:
            continue
            
        cand1_idx = random.randint(0, len(excluded) - 1)
        cand2_idx = random.randint(0, len(excluded) - 1)
        if cand1_idx == cand2_idx:
            continue
            
        cand1 = excluded[cand1_idx]
        cand2 = excluded[cand2_idx]
        
        # Check if adding cand1 and cand2 is valid
        r1 = temp_reachable | (temp_reachable << cand1)
        r1 &= MASK
        if r1 & TARGET_BIT:
            continue
            
        r2 = r1 | (r1 << cand2)
        r2 &= MASK
        if r2 & TARGET_BIT:
            continue
            
        # Valid improvement
        best_selected = temp_selected + [cand1, cand2]
        best_reachable = r2
        best_size += 1
        
        # Update excluded list
        excluded.remove(cand1)
        excluded.remove(cand2)
        excluded.append(rem_val)
        
        # If we improved, we can restart the "add if valid" phase quickly
        # But let's keep the loop going to find more improvements.
        
    # Final check
    if not best_selected:
        return [1]
        
    return best_selected
