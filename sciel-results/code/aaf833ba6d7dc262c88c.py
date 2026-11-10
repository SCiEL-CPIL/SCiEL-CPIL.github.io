import time
import random

def construct():
    N = 512
    start_time = time.time()
    time_limit = 57.0  # Safety margin before 60s

    # Precompute all 4-term arithmetic progressions using bitmask representation
    # constraints[x] = list of bitmasks representing the other 3 elements in each 4-AP containing x
    constraints = [[] for _ in range(N + 1)]
    
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            
            mask_bce = (1 << (b - 1)) | (1 << (c - 1)) | (1 << (e - 1))
            constraints[a].append(mask_bce)
            
            mask_ace = (1 << (a - 1)) | (1 << (c - 1)) | (1 << (e - 1))
            constraints[b].append(mask_ace)
            
            mask_abe = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (e - 1))
            constraints[c].append(mask_abe)
            
            mask_abc = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            constraints[e].append(mask_abc)

    def can_add(x, S_mask):
        """Check if x can be added to set represented by S_mask (bitmask)."""
        for m in constraints[x]:
            if (m & S_mask) == m:
                return False
        return True

    def can_swap(S_mask, i, j):
        """Check if swapping i (in S) with j (not in S) keeps S valid.
        i is 1-indexed element in S, j is 1-indexed element not in S."""
        i_bit = 1 << (i - 1)
        # After swap, S' = S - {i} + {j}
        # A 4-AP is formed iff it contains j and its other 3 elements are in S'
        # Other 3 elements must be in S\{i}, i.e., in S but not i
        for m in constraints[j]:
            # If i is one of the other 3 elements in this AP, skip (it's being removed)
            if m & i_bit:
                continue
            # Check if all 3 other elements are in S (and none is i, already handled)
            if (m & S_mask) == m:
                return False
        return True

    def local_search(S_mask):
        """Perform local search to maximize size of S represented as bitmask."""
        size = bin(S_mask).count('1')
        
        while time.time() - start_time < time_limit:
            added = False
            # Try to add elements in random order
            candidates = []
            for x in range(1, N + 1):
                bit = 1 << (x - 1)
                if not (S_mask & bit):
                    candidates.append(x)
            random.shuffle(candidates)
            
            for x in candidates:
                if can_add(x, S_mask):
                    S_mask |= (1 << (x - 1))
                    size += 1
                    added = True
                    break
            
            if added:
                continue
                
            # If no element could be added, try swaps
            swapped = False
            elements_in_S = [x for x in range(1, N + 1) if (S_mask >> (x - 1)) & 1]
            elements_out = [x for x in range(1, N + 1) if not ((S_mask >> (x - 1)) & 1)]
            random.shuffle(elements_in_S)
            random.shuffle(elements_out)
            
            for i in elements_in_S:
                if time.time() - start_time > time_limit:
                    break
                for j in elements_out:
                    if can_swap(S_mask, i, j):
                        S_mask &= ~(1 << (i - 1))
                        S_mask |= (1 << (j - 1))
                        swapped = True
                        break
                if swapped:
                    break
            
            if not swapped:
                # Try 2-opt: remove two, add two
                found_two_opt = False
                if len(elements_in_S) >= 2 and len(elements_out) >= 2:
                    for idx1 in range(min(len(elements_in_S), 50)):
                        if time.time() - start_time > time_limit:
                            break
                        i1 = elements_in_S[idx1]
                        for idx2 in range(idx1 + 1, min(len(elements_in_S), 50)):
                            i2 = elements_in_S[idx2]
                            for j1 in elements_out[:30]:
                                for j2 in elements_out[30:60]:
                                    if j1 == j2:
                                        continue
                                    # Check if we can remove i1, i2 and add j1, j2
                                    S_temp = S_mask
                                    S_temp &= ~(1 << (i1 - 1))
                                    S_temp &= ~(1 << (i2 - 1))
                                    if can_add(j1, S_temp) and can_add(j2, S_temp | (1 << (j1 - 1))):
                                        S_mask = S_temp | (1 << (j1 - 1)) | (1 << (j2 - 1))
                                        found_two_opt = True
                                        break
                                if found_two_opt:
                                    break
                            if found_two_opt:
                                break
                        if found_two_opt:
                            break
                
                if not found_two_opt:
                    break
                    
        return S_mask, size

    # Strategy: Multiple restarts with randomization
    best_mask = 0
    best_size = 0
    
    # Start 1: Greedy (sorted order)
    S_curr = 0
    for x in range(1, N + 1):
        if can_add(x, S_curr):
            S_curr |= (1 << (x - 1))
    mask_opt, size_opt = local_search(S_curr)
    if size_opt > best_size:
        best_mask = mask_opt
        best_size = size_opt
    
    # Start 2: Randomized Greedy
    if time.time() - start_time < time_limit - 10:
        S_curr = 0
        order = list(range(1, N + 1))
        random.shuffle(order)
        for x in order:
            if can_add(x, S_curr):
                S_curr |= (1 << (x - 1))
        mask_opt, size_opt = local_search(S_curr)
        if size_opt > best_size:
            best_mask = mask_opt
            best_size = size_opt

    # Start 3: Another Randomized Greedy
    if time.time() - start_time < time_limit - 10:
        S_curr = 0
        order = list(range(1, N + 1))
        random.shuffle(order)
        for x in order:
            if can_add(x, S_curr):
                S_curr |= (1 << (x - 1))
        mask_opt, size_opt = local_search(S_curr)
        if size_opt > best_size:
            best_mask = mask_opt
            best_size = size_opt

    # Start 4: Greedy with different randomization
    if time.time() - start_time < time_limit - 5:
        S_curr = 0
        order = list(range(1, N + 1))
        random.shuffle(order)
        for x in order:
            if can_add(x, S_curr):
                S_curr |= (1 << (x - 1))
        mask_opt, size_opt = local_search(S_curr)
        if size_opt > best_size:
            best_mask = mask_opt
            best_size = size_opt

    # Start 5: If time remains, try another
    if time.time() - start_time < time_limit - 5:
        S_curr = 0
        order = list(range(1, N + 1))
        random.shuffle(order)
        for x in order:
            if can_add(x, S_curr):
                S_curr |= (1 << (x - 1))
        mask_opt, size_opt = local_search(S_curr)
        if size_opt > best_size:
            best_mask = mask_opt
            best_size = size_opt

    # Convert bitmask to sorted list
    result = []
    for x in range(1, N + 1):
        if (best_mask >> (x - 1)) & 1:
            result.append(x)
    
    return sorted(result)
