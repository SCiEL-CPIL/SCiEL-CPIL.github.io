"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _precompute_triples(n_max=180):
    """
    Pre-compute all triples (a, b, c) with a < b < c <= n_max 
    such that 1/a = 1/b + 1/c.
    Returns:
    - triples: list of tuples (a, b, c)
    - elem_to_triples: dict mapping element -> list of indices in triples
    """
    triples = []
    elem_to_triples = {i: [] for i in range(1, n_max + 1)}
    
    for a in range(1, n_max + 1):
        # (b-a)(c-a) = a^2
        # Let x = b-a, y = c-a. xy = a^2. x < y.
        # b = a+x, c = a+y.
        # Constraints: b < c => x < y.
        # b, c <= n_max => a+x <= n_max, a+y <= n_max.
        # x must be a divisor of a^2.
        a_sq = a * a
        # Iterate divisors x of a^2 up to sqrt(a^2) = a
        # We need x < y, so x < a.
        for x in range(1, a):
            if a_sq % x == 0:
                y = a_sq // x
                # x < y is guaranteed if x < a and y = a^2/x > a
                # Check bounds
                b = a + x
                c = a + y
                if b <= n_max and c <= n_max:
                    # Ensure distinctness: a < b < c
                    # a < a+x is true for x>0
                    # a+x < a+y is true for x<y
                    idx = len(triples)
                    triples.append((a, b, c))
                    elem_to_triples[a].append(idx)
                    elem_to_triples[b].append(idx)
                    elem_to_triples[c].append(idx)
                    
    return triples, elem_to_triples

def construct():
    start_time = time.time()
    deadline = 58.0 # Safety margin
    
    n_max = 180
    all_elements = set(range(1, n_max + 1))
    triples, elem_to_triples = _precompute_triples(n_max)
    num_triples = len(triples)
    
    if num_triples == 0:
        return list(all_elements)
        
    best_set = set()
    best_size = 0
    
    # Multi-restart greedy
    # Number of restarts depends on time. Let's try to run as many as possible.
    # Each restart:
    # 1. Init S = all_elements
    # 2. Compute initial conflict counts
    # 3. Greedy removal
    # 4. Add-back
    # 5. Update best
    
    # We can estimate cost per restart.
    # Removal: O(N * E_avg)
    # Add-back: O(N * E_avg)
    # N=180. E_avg is small.
    
    restarts = 0
    while time.time() - start_time < deadline - 5.0: # Leave 5s for final local search
        restarts += 1
        random.seed(42 + restarts) # Reproducible but varied
        
        S = set(all_elements)
        
        # State for conflict tracking
        # triple_state[i]: number of elements of triple i currently in S
        triple_state = [3] * num_triples # Initially all 3 are in S
        # conflict_count[x]: number of triples where all 3 elements are in S, involving x
        conflict_count = {i: 0 for i in range(1, n_max + 1)}
        
        # Initialize conflict_count
        for idx in range(num_triples):
            a, b, c = triples[idx]
            conflict_count[a] += 1
            conflict_count[b] += 1
            conflict_count[c] += 1
            
        total_conflicts = num_triples
        
        # Greedy Removal
        while total_conflicts > 0:
            # Find element with max conflict count in S
            max_count = 0
            candidates = []
            for x in S:
                cnt = conflict_count[x]
                if cnt > max_count:
                    max_count = cnt
                    candidates = [x]
                elif cnt == max_count and cnt > 0:
                    candidates.append(x)
                    
            if not candidates:
                break
                
            # Random tie-breaking
            v = random.choice(candidates)
            S.remove(v)
            
            # Update states
            for tidx in elem_to_triples[v]:
                old_state = triple_state[tidx]
                if old_state == 3:
                    # This triple was a conflict, now it's not
                    total_conflicts -= 1
                    a, b, c = triples[tidx]
                    # Decrement conflict count for the other two
                    for u in (a, b, c):
                        if u != v:
                            conflict_count[u] -= 1
                triple_state[tidx] = old_state - 1
                
        # Add-back Phase
        # Try to add back elements not in S
        # Order: Random or by original conflict count? Random is good for diversity.
        removed_elements = list(all_elements - S)
        random.shuffle(removed_elements)
        
        for x in removed_elements:
            # Check if x can be added
            # x can be added if for all triples (x, u, v), NOT (u in S and v in S)
            can_add = True
            for tidx in elem_to_triples[x]:
                a, b, c = triples[tidx]
                # The other two elements
                if x == a:
                    u, v = b, c
                elif x == b:
                    u, v = a, c
                else:
                    u, v = a, b
                    
                if u in S and v in S:
                    can_add = False
                    break
            
            if can_add:
                S.add(x)
                # Update states? Not strictly necessary for validity check of next adds 
                # if we just check membership, but for efficiency in subsequent adds?
                # Since we are just checking membership in S, and S is a set, 
                # the check `u in S and v in S` is O(1).
                # We don't need to update triple_state for the add-back phase 
                # because we are not removing anything else.
                
        # Update best
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            
    # Local Search on best_set
    # Try 2-for-1 swaps: Remove y, add x1, x2?
    # Or simpler: Remove y, see if we can add MORE elements than 1?
    # Actually, just try to add elements that are currently blocked.
    # A blocked element x is blocked by some pair (u,v) in S.
    # If we remove u, x might become addable.
    
    # Let's do a limited local search:
    # For each y in best_set:
    #   Remove y
    #   Try to add all possible x not in S
    #   If size increases, keep it.
    #   Else, put y back.
    
    if time.time() - start_time < deadline - 1.0:
        current_best = set(best_set)
        improved = True
        while improved and time.time() - start_time < deadline - 1.0:
            improved = False
            # Iterate over elements to remove
            # To avoid bias, shuffle
            elems_to_try = list(current_best)
            random.shuffle(elems_to_try)
            
            for y in elems_to_try:
                if time.time() - start_time > deadline - 1.0:
                    break
                    
                # Remove y
                current_best.remove(y)
                original_size = len(current_best)
                
                # Try to add back all elements not in current_best
                # Include y? No, y was removed. We want to see if removing y allows adding others.
                # We should also try to add y back if it's valid, but that just restores state.
                # We want to add OTHERS.
                
                candidates = [x for x in range(1, n_max + 1) if x not in current_best]
                random.shuffle(candidates)
                
                added_any = False
                for x in candidates:
                    can_add = True
                    for tidx in elem_to_triples[x]:
                        a, b, c = triples[tidx]
                        if x == a:
                            u, v = b, c
                        elif x == b:
                            u, v = a, c
                        else:
                            u, v = a, b
                            
                        if u in current_best and v in current_best:
                            can_add = False
                            break
                            
                    if can_add:
                        current_best.add(x)
                        added_any = True
                        
                # Check if we improved size
                if len(current_best) > best_size:
                    best_size = len(current_best)
                    best_set = set(current_best)
                    improved = True
                    # Restart the local search loop with the new best? 
                    # Or just continue?
                    # Let's break and restart the while loop to shuffle again
                    break
                else:
                    # Restore y if we didn't improve
                    # Actually, if we didn't improve, current_best might be same size or smaller.
                    # We want to keep the best found.
                    # If len(current_best) < best_size, we should revert.
                    # But we are modifying current_best.
                    # Let's revert to the state before removing y if no improvement.
                    # Wait, if we added elements, size might be same.
                    # If size is same, we might have a different set of same size.
                    # The task is to maximize cardinality. Same size is not an improvement.
                    # So we revert.
                    current_best = set(best_set) # Revert to last best
                    # Note: This is inefficient if we do it for every y.
                    # Better: Keep a copy of the set before removing y.
                    # But copying 150 elements is fast.
                    
                    # Let's refine the local search logic to be safer.
                    # Actually, the simple add-back in the main loop is quite effective.
                    # The local search might be overkill or risky.
                    # Let's stick to the multi-restart greedy which is robust.
                    break # Break out of y loop if no improvement from this y
                    
    return list(best_set)

# EVOLVE-BLOCK-END
