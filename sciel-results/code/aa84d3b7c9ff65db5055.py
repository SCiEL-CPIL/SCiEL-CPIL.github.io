import time
import random

def construct():
    N = 512
    start_time = time.time()
    time_limit = 85.0  # Safety margin before 90s limit

    # Precompute all 4-term arithmetic progressions
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

    def is_valid_swap_remove(i, S_mask):
        """
        Check if removing i from S_mask results in a valid set (which it always does, 
        as removing elements cannot create new APs). 
        This is trivially true.
        """
        return True

    def can_add_after_remove(x, S_mask, removed_i):
        """
        Check if x can be added to S_mask \ {removed_i}.
        """
        S_temp = S_mask & ~(1 << (removed_i - 1))
        for m in constraints[x]:
            if (m & S_temp) == m:
                return False
        return True

    def local_search(S_mask):
        """
        Perform local search to maximize size of S represented as bitmask.
        Uses Add, Expand (remove 1, add 2), and Perturb (swap 1 for 1).
        """
        size = bin(S_mask).count('1')
        
        # Precompute list of elements for faster iteration
        all_elements = list(range(1, N + 1))
        
        while time.time() - start_time < time_limit:
            added = False
            expanded = False
            perturbed = False
            
            # 1. Try to add a random element
            random.shuffle(all_elements)
            for x in all_elements:
                bit = 1 << (x - 1)
                if not (S_mask & bit):
                    if can_add(x, S_mask):
                        S_mask |= bit
                        size += 1
                        added = True
                        break
            
            if added:
                continue
            
            # 2. Try to Expand: Remove 1, Add 2 (Net +1 size)
            # Sample a few candidates for removal
            elements_in_S = [x for x in range(1, N + 1) if (S_mask >> (x - 1)) & 1]
            elements_out = [x for x in range(1, N + 1) if not ((S_mask >> (x - 1)) & 1)]
            
            if elements_in_S and len(elements_out) >= 2:
                # Randomly select a few removal candidates
                random.shuffle(elements_in_S)
                removal_candidates = elements_in_S[:10] # Limit to 10 to save time
                
                for i in removal_candidates:
                    if time.time() - start_time > time_limit:
                        break
                    
                    S_temp = S_mask & ~(1 << (i - 1))
                    
                    # Try to find two elements j, k to add
                    # We can iterate through out elements
                    # Optimization: If we add j, we must check k against S_temp | j
                    found_pair = False
                    # Shuffle out elements to randomize
                    out_list = elements_out[:]
                    random.shuffle(out_list)
                    
                    for j in out_list:
                        if found_pair:
                            break
                        if not can_add_after_remove(j, S_mask, i):
                            continue
                        
                        S_temp_j = S_temp | (1 << (j - 1))
                        
                        for k in out_list:
                            if k == j:
                                continue
                            # Check if k can be added to S_temp_j
                            # Note: k is not in S_mask, so not in S_temp or S_temp_j (unless j==k which is handled)
                            # But wait, if j and k are both in out_list, they are distinct.
                            # We need to check if k forms AP with S_temp_j
                            if can_add(k, S_temp_j):
                                # Valid expansion
                                S_mask = S_temp_j | (1 << (k - 1))
                                size += 1
                                expanded = True
                                found_pair = True
                                break
                            if time.time() - start_time > time_limit:
                                break
                    
                    if expanded:
                        break
            
            if expanded:
                continue

            # 3. Try to Perturb: Swap 1 for 1 (Net 0 size) to escape local optimum
            if elements_in_S and elements_out:
                random.shuffle(elements_in_S)
                random.shuffle(elements_out)
                
                # Try a few swaps
                for i in elements_in_S[:5]:
                    if time.time() - start_time > time_limit:
                        break
                    for j in elements_out[:5]:
                        # Check if swap i->j is valid
                        # New set: S_mask - i + j
                        # Valid if no 4-AP contains j and its other 3 are in S_mask - i
                        valid_swap = True
                        i_bit = 1 << (i - 1)
                        for m in constraints[j]:
                            # If i is in m, it's removed, so check rest against S_mask
                            # If i is not in m, check all against S_mask
                            if m & i_bit:
                                # m contains i. The other 3 elements must be in S_mask (which they are, as m is AP with j)
                                # Wait, if m contains i, then m \ {i} must be in S_mask.
                                # Since i is in S_mask, (m & S_mask) will have i.
                                # We need (m & (S_mask ^ i_bit)) == (m ^ i_bit)
                                if (m & (S_mask ^ i_bit)) != (m ^ i_bit):
                                    valid_swap = False
                                    break
                            else:
                                # m does not contain i. All 3 elements in m must be in S_mask.
                                if (m & S_mask) != m:
                                    valid_swap = False
                                    break
                        
                        if valid_swap:
                            S_mask = (S_mask ^ i_bit) | (1 << (j - 1))
                            perturbed = True
                            break
                    if perturbed:
                        break
            
            if not perturbed:
                # Stuck. Break out to restart with new random seed
                break
                
        return S_mask, size

    # Strategy: Multiple restarts
    best_mask = 0
    best_size = 0
    
    # Restart 1: Greedy (sorted)
    S_curr = 0
    for x in range(1, N + 1):
        if can_add(x, S_curr):
            S_curr |= (1 << (x - 1))
    mask_opt, size_opt = local_search(S_curr)
    if size_opt > best_size:
        best_mask = mask_opt
        best_size = size_opt

    # Restart 2-10: Randomized Greedy
    for _ in range(9):
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
