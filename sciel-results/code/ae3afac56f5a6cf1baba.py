"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    from math import gcd
    from fractions import Fraction
    
    # Set a safety deadline. The evaluator kills at 90s, but we want to return well before.
    deadline = time.time() + 55.0
    
    nums = list(range(2, 257))
    n = len(nums)
    
    # Precompute conflicts using bitmasks for speed
    # conflict_mask[i] is a bitmask of indices j such that lcm(nums[i], nums[j]) <= 256
    conflict_mask = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            g = gcd(nums[i], nums[j])
            if (nums[i] // g) * nums[j] <= 256:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    # Precompute reciprocal values as Fractions for exact arithmetic
    recip = [Fraction(1, nums[i]) for i in range(n)]
    
    def is_valid_add(selected_mask, i):
        """Check if adding i to selected is valid (no conflicts)."""
        return (conflict_mask[i] & selected_mask) == 0
    
    def mask_to_set(mask):
        return set(i for i in range(n) if (mask >> i) & 1)
    
    def set_to_mask(s):
        m = 0
        for i in s:
            m |= (1 << i)
        return m
    
    def local_search(init_mask, max_iters=30):
        """
        Local search starting from init_mask.
        Returns (best_mask, best_score_fraction).
        """
        best = init_mask
        best_score = sum(recip[i] for i in range(n) if (best >> i) & 1)
        best_list = mask_to_set(best)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            improved = False
            
            # 1. Try adding one element
            best_add = -1
            best_add_val = Fraction(0)
            for i in range(n):
                if (best >> i) & 1:
                    continue
                if is_valid_add(best, i):
                    val = recip[i]
                    if val > best_add_val:
                        best_add_val = val
                        best_add = i
            
            if best_add != -1:
                best |= (1 << best_add)
                best_score += best_add_val
                best_list.add(best_add)
                improved = True
            
            # 2. Try remove-one-add-one (swap)
            if time.time() < deadline - 5.0:
                best_swap = None
                best_swap_gain = Fraction(0)
                
                for i in best_list:
                    if time.time() > deadline:
                        break
                    for j in range(n):
                        if (best >> j) & 1:
                            continue
                        if nums[j] >= nums[i]:
                            continue
                        
                        # Check if j conflicts with any element in best except possibly i
                        conflicts_with_best = conflict_mask[j] & best
                        if conflicts_with_best:
                            # j can only conflict with i (the element being removed)
                            if conflicts_with_best != (1 << i):
                                continue
                        
                        gain = recip[j] - recip[i]
                        if gain > best_swap_gain:
                            best_swap_gain = gain
                            best_swap = (i, j)
                            
                if best_swap is not None:
                    i, j = best_swap
                    best &= ~(1 << i)
                    best |= (1 << j)
                    best_score += best_swap_gain
                    best_list.discard(i)
                    best_list.add(j)
                    improved = True
            
            # 3. Try 2-for-2 swap (remove two, add two)
            if time.time() < deadline - 3.0 and len(best_list) >= 2:
                best_2swap = None
                best_2swap_gain = Fraction(0)
                
                best_list_sorted = sorted(best_list)
                for idx1 in range(len(best_list_sorted)):
                    if time.time() > deadline:
                        break
                    i = best_list_sorted[idx1]
                    for idx2 in range(idx1+1, len(best_list_sorted)):
                        i2 = best_list_sorted[idx2]
                        
                        # Try to find two elements to add
                        for j1 in range(n):
                            if (best >> j1) & 1:
                                continue
                            if nums[j1] >= nums[i]:
                                continue
                            
                            # Check j1 conflicts with best except i and i2
                            conflicts_j1 = conflict_mask[j1] & best
                            allowed_conflicts = (1 << i) | (1 << i2)
                            if conflicts_j1 & ~allowed_conflicts:
                                continue
                            
                            for j2 in range(n):
                                if (best >> j2) & 1:
                                    continue
                                if j2 == j1:
                                    continue
                                if nums[j2] >= nums[i2]:
                                    continue
                                
                                # Check j2 conflicts with best except i and i2
                                conflicts_j2 = conflict_mask[j2] & best
                                if conflicts_j2 & ~allowed_conflicts:
                                    continue
                                
                                # Check j1 and j2 don't conflict with each other
                                if (conflict_mask[j1] >> j2) & 1:
                                    continue
                                
                                gain = recip[j1] + recip[j2] - recip[i] - recip[i2]
                                if gain > best_2swap_gain:
                                    best_2swap_gain = gain
                                    best_2swap = (i, i2, j1, j2)
                        
                        if time.time() > deadline:
                            break
                
                if best_2swap is not None:
                    i, i2, j1, j2 = best_2swap
                    best &= ~(1 << i)
                    best &= ~(1 << i2)
                    best |= (1 << j1)
                    best |= (1 << j2)
                    best_score += best_2swap_gain
                    best_list.discard(i)
                    best_list.discard(i2)
                    best_list.add(j1)
                    best_list.add(j2)
                    improved = True
            
            if not improved:
                break
        
        return best, best_score

    def greedy_construct(exclude_mask):
        """
        Construct a valid set by greedily adding smallest available numbers
        not in exclude_mask.
        """
        current_mask = 0
        for i in range(n):
            if (exclude_mask >> i) & 1:
                continue
            if is_valid_add(current_mask, i):
                current_mask |= (1 << i)
        return current_mask

    best_mask = 0
    best_score = Fraction(0)
    
    # 1. Standard Greedy (no exclusions)
    init_mask = greedy_construct(0)
    ls_mask, ls_score = local_search(init_mask, max_iters=15)
    if ls_score > best_score:
        best_mask = ls_mask
        best_score = ls_score
        
    # 2. Exclusion Prefixes
    max_prefix = 25
    for p in range(1, max_prefix + 1):
        if time.time() > deadline - 10.0:
            break
        exclude_mask = 0
        for i in range(p):
            exclude_mask |= (1 << i)
        init_mask = greedy_construct(exclude_mask)
        ls_mask, ls_score = local_search(init_mask, max_iters=15)
        if ls_score > best_score:
            best_mask = ls_mask
            best_score = ls_score

    # 3. Random Restarts with remaining time
    random.seed(42)
    while time.time() < deadline - 2.0:
        # Generate a random start
        if random.random() < 0.5:
            # Random subset
            s = random.sample(range(n), 50)
            current_mask = 0
            for i in sorted(s, key=lambda x: nums[x]):
                if is_valid_add(current_mask, i):
                    current_mask |= (1 << i)
        else:
            # Perturb best
            best_list = mask_to_set(best_mask)
            current_mask = best_mask
            for _ in range(5):
                if best_list:
                    victim = random.choice(list(best_list))
                    current_mask &= ~(1 << victim)
                    best_list.discard(victim)
        
        ls_mask, ls_score = local_search(current_mask, max_iters=8)
        if ls_score > best_score:
            best_mask = ls_mask
            best_score = ls_score
            
    return [nums[i] for i in range(n) if (best_mask >> i) & 1]
# EVOLVE-BLOCK-END
