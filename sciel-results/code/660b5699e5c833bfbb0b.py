import time
import math

def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 145.0  # 145s limit to be safe

    # 1. Find all divisors of 51975
    divisors = []
    for i in range(1, int(math.isqrt(base)) + 1):
        if base % i == 0:
            if i > 1:
                divisors.append(i)
            other = base // i
            if other > 1 and other != i:
                divisors.append(other)
    divisors.sort()
    
    # Map modulus to index for faster lookup if needed, but list is small
    # divisors list contains all m > 1 dividing base
    
    # 2. Precompute masks for each (m, r)
    # masks[m][r] is an integer bitset
    masks = {}
    for m in divisors:
        masks[m] = [0] * m
        for r in range(m):
            mask_val = 0
            for x in range(r, base, m):
                mask_val |= (1 << x)
            masks[m][r] = mask_val
            
    # Initial state
    uncovered_mask = (1 << base) - 1
    used_moduli = set()
    classes = []
    freq = [0] * base
    unique_mask = 0  # Bitset of residues with freq == 1
    
    def add_class(m, r):
        nonlocal uncovered_mask, unique_mask
        mask = masks[m][r]
        # Update freq and unique_mask
        # We need to iterate through set bits of mask
        # Or just iterate through range(r, base, m)
        for x in range(r, base, m):
            if freq[x] == 0:
                uncovered_mask &= ~(1 << x)
                # freq was 0, now 1. Add to unique_mask
                unique_mask |= (1 << x)
            elif freq[x] == 1:
                # freq was 1, now 2. Remove from unique_mask
                unique_mask &= ~(1 << x)
            freq[x] += 1
        used_moduli.add(m)
        classes.append([m, r])

    def remove_class(m, r):
        nonlocal uncovered_mask, unique_mask
        mask = masks[m][r]
        for x in range(r, base, m):
            freq[x] -= 1
            if freq[x] == 1:
                # freq was 2, now 1. Add to unique_mask
                unique_mask |= (1 << x)
            elif freq[x] == 0:
                # freq was 1, now 0. Remove from unique_mask, add to uncovered
                unique_mask &= ~(1 << x)
                uncovered_mask |= (1 << x)
        used_moduli.remove(m)
        # Find and remove from classes list
        for i, (cm, cr) in enumerate(classes):
            if cm == m and cr == r:
                classes.pop(i)
                break

    # 3. Greedy Set Cover
    remaining_moduli = set(divisors)
    
    while remaining_moduli:
        best_gain = 0
        best_m = None
        best_r = None
        
        for m in remaining_moduli:
            for r in range(m):
                gain = (uncovered_mask & masks[m][r]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_m = m
                    best_r = r
                    
        if best_gain == 0:
            break
            
        add_class(best_m, best_r)
        remaining_moduli.remove(best_m)
        
        if time.time() > deadline:
            break

    # 4. Local Search
    # We try swaps: remove (m_old, r_old), add (m_new, r_new)
    # m_new must be in remaining_moduli (unused)
    # m_old must be in used_moduli
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_swap_gain = 0
        best_old_m = None
        best_old_r = None
        best_new_m = None
        best_new_r = None
        
        # Iterate over all used classes
        for m_old, r_old in classes:
            # Calculate loss part: mask_old & unique_mask
            # This represents residues that would become uncovered if we remove this class
            # We don't need to calculate popcount yet, just keep the mask
            # Actually, we need to evaluate swaps.
            # For each unused modulus m_new, find best r_new
            for m_new in remaining_moduli:
                # For this pair (m_old, m_new), find r_new that maximizes net gain
                # Net gain = popcount(mask_new & (uncovered_mask | (mask_old & unique_mask))) - popcount(mask_old & unique_mask)
                # Let U_expanded = uncovered_mask | (masks[m_old][r_old] & unique_mask)
                # We want to maximize popcount(masks[m_new][r_new] & U_expanded)
                # The term popcount(mask_old & unique_mask) is constant for this m_old
                # So for each m_new, we just find r_new maximizing intersection with U_expanded
                
                # Optimization: If m_new is very large, iterating all r might be slow?
                # Max m is 51975. 51975 iterations * (num_used * num_unused)
                # Num used <= 48, Num unused <= 48.
                # 48 * 48 * 51975 ~ 120 million ops. This might be too slow in Python if done repeatedly.
                
                # Let's check time budget.
                # We should limit the search or optimize.
                
                # Alternative: Only try swaps if potential gain is high.
                # Or, limit the number of swaps per iteration.
                
                # Let's try a simpler heuristic first:
                # For each unused m_new, find the best r_new against current uncovered_mask.
                # Then check if swapping with any used class improves things.
                
                # Actually, let's just iterate. 120M simple bit operations might take a few seconds in PyPy, but CPython might be slower.
                # Let's add a time check inside the inner loop.
                
                u_expanded = uncovered_mask | (masks[m_old][r_old] & unique_mask)
                # Find best r_new for m_new
                best_r_new = 0
                best_int_count = 0
                for r_new in range(m_new):
                    cnt = (masks[m_new][r_new] & u_expanded).bit_count()
                    if cnt > best_int_count:
                        best_int_count = cnt
                        best_r_new = r_new
                
                # Calculate net gain
                # loss = popcount(masks[m_old][r_old] & unique_mask)
                loss = (masks[m_old][r_old] & unique_mask).bit_count()
                net_gain = best_int_count - loss
                
                if net_gain > best_swap_gain:
                    best_swap_gain = net_gain
                    best_old_m = m_old
                    best_old_r = r_old
                    best_new_m = m_new
                    best_new_r = r_new
                    
                if time.time() > deadline:
                    break
                    
            if time.time() > deadline:
                break
                
        if best_swap_gain > 0:
            # Perform swap
            remove_class(best_old_m, best_old_r)
            add_class(best_new_m, best_new_r)
            improved = True
            remaining_moduli.add(best_old_m)
            remaining_moduli.remove(best_new_m)
            
        if time.time() > deadline:
            break

    return classes
