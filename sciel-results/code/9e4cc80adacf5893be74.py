"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def construct():
    t0 = time.time()
    deadline = t0 + 80.0  # safety margin before 90s kill
    
    N = 256
    # Conflict matrix: conflict[i][j] = True if (i+1)*(j+1) % (i+1+j+1) == 0
    # Using 0-indexed: element i represents number i+1
    # Precompute conflicts as bitmasks for each element
    # conflict_mask[i] is a bitmask where bit j is set if i and j conflict
    
    conflict_mask = [0] * N
    for i in range(N):
        a = i + 1
        mask = 0
        for j in range(N):
            if i == j:
                continue
            b = j + 1
            if (a * b) % (a + b) == 0:
                mask |= (1 << j)
        conflict_mask[i] = mask
    
    # Helper: check if adding element x to a set (represented as bitmask) creates conflicts
    # An element x conflicts with set S if conflict_mask[x] & S != 0
    # But we need: for all y in S, (x+1)*(y+1) % (x+1+y+1) != 0
    # i.e., conflict_mask[x] & S == 0
    
    # Initial solution: all odd numbers (indices 0, 2, 4, ..., 254 -> numbers 1, 3, 5, ..., 255)
    # Verify this is valid: for two odd numbers a,b, a+b is even, a*b is odd
    # (a+b) | (a*b) requires a+b | a*b. Since a+b is even and a*b is odd, 
    # a+b cannot divide a*b unless a+b = 1, which is impossible.
    # Wait, that's not quite right. Let me reconsider.
    # Actually for odd a, b: a+b is even. a*b is odd. An even number cannot divide an odd number.
    # So yes, all odds is a valid set.
    
    best_set = 0
    for i in range(0, N, 2):  # even indices = odd numbers 1,3,5,...,255
        best_set |= (1 << i)
    
    best_size = best_set.bit_count()
    
    # Local search: try to add elements, remove conflicting pairs, do swaps
    # Phase 1: Greedy addition - try to add each element if it doesn't conflict
    for i in range(N):
        if (best_set >> i) & 1:
            continue
        # Check if i can be added
        if (conflict_mask[i] & best_set) == 0:
            best_set |= (1 << i)
            best_size += 1
    
    # Phase 2: Local search with removals and swaps
    # Try removing one element and adding two others (net +1)
    # Or removing one and adding one (net 0, but might enable future additions)
    
    # Precompute for each element, the set of elements it conflicts with (as list for iteration)
    # conflict_list[i] = list of j where conflict_mask[i] has bit j set
    conflict_list = []
    for i in range(N):
        lst = []
        m = conflict_mask[i]
        while m:
            low = m & (-m)
            j = low.bit_length() - 1
            lst.append(j)
            m ^= low
        conflict_list[i] = lst
    
    # Local search loop
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding any single element
        for i in range(N):
            if (best_set >> i) & 1:
                continue
            if (conflict_mask[i] & best_set) == 0:
                best_set |= (1 << i)
                best_size += 1
                improved = True
        
        # Try 1-for-2 swaps: remove element r, add elements a and b
        # This is expensive, so do a limited version
        if time.time() < deadline:
            # Try removing each element and see if we can add 2 back
            for r in range(N):
                if not ((best_set >> r) & 1):
                    continue
                # Try removing r
                new_set = best_set & ~(1 << r)
                # Find elements that can be added to new_set
                addable = []
                for i in range(N):
                    if (new_set >> i) & 1:
                        continue
                    if (conflict_mask[i] & new_set) == 0:
                        addable.append(i)
                
                if len(addable) >= 2:
                    # Try adding pairs from addable
                    for idx_a in range(len(addable)):
                        a = addable[idx_a]
                        for idx_b in range(idx_a + 1, len(addable)):
                            b = addable[idx_b]
                            # Check a and b don't conflict with each other
                            if (conflict_mask[a] >> b) & 1:
                                continue
                            new_set2 = new_set | (1 << a) | (1 << b)
                            # Verify no conflicts
                            valid = True
                            # a and b are already checked against new_set and each other
                            # But need to verify: the new_set might have elements that conflict with a or b
                            # Actually we checked conflict_mask[a] & new_set == 0 and conflict_mask[b] & new_set == 0
                            # And conflict_mask[a] & (1<<b) == 0 and conflict_mask[b] & (1<<a) == 0
                            # So it should be valid
                            if new_set2.bit_count() > best_size:
                                best_set = new_set2
                                best_size = best_set.bit_count()
                                improved = True
                                break
                        if improved:
                            break
                if improved:
                    break
        
        # Try 1-for-1 swaps that might enable future improvements
        if time.time() < deadline:
            for r in range(N):
                if not ((best_set >> r) & 1):
                    continue
                new_set = best_set & ~(1 << r)
                # Find elements that can be added
                for i in range(N):
                    if (new_set >> i) & 1:
                        continue
                    if (conflict_mask[i] & new_set) == 0:
                        # This swap keeps size same, but might help later
                        # Only do if it seems beneficial (hard to tell)
                        pass
    
    # Phase 3: More aggressive local search - try removing 2, adding 3
    if time.time() < deadline:
        for _ in range(100):  # limited attempts
            if time.time() > deadline:
                break
            # Randomly pick 2 elements to remove
            elements_in_set = []
            for i in range(N):
                if (best_set >> i) & 1:
                    elements_in_set.append(i)
            
            if len(elements_in_set) < 2:
                break
            
            r1, r2 = random.sample(elements_in_set, 2)
            new_set = best_set & ~(1 << r1) & ~(1 << r2)
            
            # Find all addable elements
            addable = []
            for i in range(N):
                if (new_set >> i) & 1:
                    continue
                if (conflict_mask[i] & new_set) == 0:
                    addable.append(i)
            
            if len(addable) < 3:
                continue
            
            # Try to add 3 non-conflicting elements from addable
            found = False
            for ia in range(len(addable)):
                a = addable[ia]
                for ib in range(ia + 1, len(addable)):
                    b = addable[ib]
                    if (conflict_mask[a] >> b) & 1:
                        continue
                    for ic in range(ib + 1, len(addable)):
                        c = addable[ic]
                        if (conflict_mask[a] >> c) & 1 or (conflict_mask[b] >> c) & 1:
                            continue
                        new_set2 = new_set | (1 << a) | (1 << b) | (1 << c)
                        if new_set2.bit_count() > best_size:
                            best_set = new_set2
                            best_size = best_set.bit_count()
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
    
    # Convert bitmask to list
    result = []
    for i in range(N):
        if (best_set >> i) & 1:
            result.append(i + 1)
    
    return result

# EVOLVE-BLOCK-END
