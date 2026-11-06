"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import math

    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s limit

    denominators = list(range(2, 49))  # [2, 3, ..., 48]
    n = len(denominators)

    # Calculate LCM of 2..48
    lcm_val = 1
    for d in range(2, 49):
        lcm_val = lcm_val * d // math.gcd(lcm_val, d)
    
    # Scaled values: v_i = LCM / d_i
    scaled_vals = [lcm_val // d for d in denominators]
    
    # Split into two halves
    mid = n // 2
    left_vals = scaled_vals[:mid]
    right_vals = scaled_vals[mid:]
    
    def generate_subset_sums(vals):
        sums = [0]
        for v in vals:
            # Use list comprehension for speed
            # This creates a new list of same size, then extends
            sums += [s + v for s in sums]
        return sums

    # Generate sums for left half
    left_sums = generate_subset_sums(left_vals)
    left_sums.sort()
    
    # Generate sums for right half
    right_sums = generate_subset_sums(right_vals)
    right_sums.sort()
    
    # Two-pointer search for max sum < LCM
    best_sum = 0
    best_left_idx = -1
    best_right_idx = -1
    
    # j points to the largest index in left_sums
    # We iterate through right_sums in ascending order
    j = len(left_sums) - 1
    
    for i, rs in enumerate(right_sums):
        # We need ls such that ls + rs < lcm_val
        # ls < lcm_val - rs
        target = lcm_val - rs
        while j >= 0 and left_sums[j] >= target:
            j -= 1
        if j >= 0:
            candidate_sum = left_sums[j] + rs
            if candidate_sum > best_sum:
                best_sum = candidate_sum
                best_left_idx = j
                best_right_idx = i
        # If j becomes -1, no more left sums can pair with larger right sums
        # because right sums are increasing, so target is decreasing.
        # Wait, if j becomes -1, it means left_sums[0] >= target.
        # Since target decreases as rs increases, left_sums[0] will always be >= target for subsequent i.
        # So we can break.
        if j < 0:
            break

    # Reconstruct indices
    # We need to find which bits were set for best_left_idx and best_right_idx
    # Since we generated sums by doubling, the index in the sorted list does NOT directly correspond to the bitmask.
    # We lost the mapping between sum value and bitmask during sorting!
    
    # This is a problem. We need to map sum -> bitmask.
    # Since sums might not be unique, we can just pick any bitmask that produces the sum.
    # But we need to know WHICH subset produced left_sums[best_left_idx].
    
    # Alternative: Store (sum, mask) tuples.
    # Sorting tuples is slower and uses more memory.
    # Can we recover the mask?
    # No, not easily from the sorted list of sums alone.
    
    # We need to store the mask.
    # Let's modify generate_subset_sums to return list of (sum, mask).
    
    def generate_subset_sums_with_mask(vals):
        # Returns list of (sum, mask)
        sums_masks = [(0, 0)]
        for k, v in enumerate(vals):
            bit = 1 << k
            # Create new entries
            new_entries = [(s + v, m | bit) for s, m in sums_masks]
            sums_masks.extend(new_entries)
        return sums_masks

    # This doubles memory and time.
    # 16M tuples of (int, int).
    # Tuple overhead is significant.
    
    # Let's try to be smarter.
    # We can store the sums in a list, and the masks in a separate list?
    # No, they get shuffled.
    
    # Is there a way to avoid storing masks?
    # If we just need the indices, we can recompute the mask from the sum?
    # No, many masks can have the same sum.
    
    # Let's try the tuple approach and hope it fits in time/memory.
    # 24 items -> 16M tuples.
    # Time to generate: ~2x slower than ints.
    # Time to sort: ~2x slower.
    # Total time might be 30-40s.
    # Risky.
    
    # Alternative:
    # Use the integer sums for the two-pointer search to find the best SUM.
    # Then, find a subset that achieves that sum.
    # This is a subset sum problem with a specific target.
    # We can solve it with DP or backtracking.
    # But the target is huge (LCM).
    
    # Wait, we can just store the masks for the LEFT half only?
    # No, we need both.
    
    # Let's try a different split.
    # 20 and 27?
    # 2^20 = 1M. 2^27 = 134M. Too big.
    # 22 and 25?
    # 2^22 = 4M. 2^25 = 33M.
    
    # Let's stick to 23/24 and use tuples.
    # If it's too slow, we fall back to heuristic.
    
    try:
        left_sm = generate_subset_sums_with_mask(left_vals)
        left_sm.sort(key=lambda x: x[0])
        
        right_sm = generate_subset_sums_with_mask(right_vals)
        right_sm.sort(key=lambda x: x[0])
        
        # Extract sums for binary search / two pointer
        left_sums_only = [x[0] for x in left_sm]
        right_sums_only = [x[0] for x in right_sm]
        
        best_sum = 0
        best_left_entry = None
        best_right_entry = None
        
        j = len(left_sums_only) - 1
        for i, rs in enumerate(right_sums_only):
            target = lcm_val - rs
            while j >= 0 and left_sums_only[j] >= target:
                j -= 1
            if j >= 0:
                candidate_sum = left_sums_only[j] + rs
                if candidate_sum > best_sum:
                    best_sum = candidate_sum
                    best_left_entry = left_sm[j]
                    best_right_entry = right_sm[i]
            if j < 0:
                break
        
        if best_left_entry is not None and best_right_entry is not None:
            left_mask = best_left_entry[1]
            right_mask = best_right_entry[1]
            
            indices = []
            # Left half indices are 0..mid-1
            for k in range(len(left_vals)):
                if left_mask & (1 << k):
                    indices.append(k)
            # Right half indices are mid..n-1
            for k in range(len(right_vals)):
                if right_mask & (1 << k):
                    indices.append(mid + k)
            
            return indices

    except Exception:
        pass

    # Fallback to heuristic if MITM fails or is too slow
    from fractions import Fraction
    unit_vals = [Fraction(1, d) for d in denominators]
    
    def greedy(order):
        selected = []
        total = Fraction(0)
        for idx in order:
            v = unit_vals[idx]
            if total + v < 1:
                selected.append(idx)
                total += v
        return selected, total

    best_indices = []
    best_total = Fraction(0)
    best_deficit = Fraction(1)
    
    def check_solution(indices, total):
        nonlocal best_indices, best_total, best_deficit
        if indices and total < 1:
            deficit = 1 - total
            if deficit < best_deficit:
                best_deficit = deficit
                best_total = total
                best_indices = sorted(indices)

    order_asc = list(range(n))
    order_desc = list(reversed(range(n)))
    
    for order in [order_asc, order_desc]:
        idxs, total = greedy(order)
        check_solution(idxs, total)
        
    def local_search(start_indices, max_iters=5000):
        nonlocal best_indices, best_total, best_deficit
        if not start_indices:
            return
        cur_set = set(start_indices)
        cur_total = sum(unit_vals[i] for i in cur_set)
        cur_deficit = 1 - cur_total
        if cur_deficit <= 0:
            return
        in_set = set(cur_set)
        not_in_set = set(range(n)) - cur_set
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            best_add_idx = None
            best_add_def = cur_deficit
            for add in not_in_set:
                nt = cur_total + unit_vals[add]
                if nt < 1:
                    nd = 1 - nt
                    if nd < best_add_def:
                        best_add_def = nd
                        best_add_idx = add
            if best_add_idx is not None:
                cur_set.add(best_add_idx)
                cur_total += unit_vals[best_add_idx]
                cur_deficit = best_add_def
                not_in_set.discard(best_add_idx)
                in_set.add(best_add_idx)
                improved = True
                check_solution(sorted(cur_set), cur_total)
                
            if time.time() < deadline - 5:
                best_swap_rm = None
                best_swap_add = None
                best_swap_def = cur_deficit
                for rm in list(in_set):
                    v_rm = unit_vals[rm]
                    base_total = cur_total - v_rm
                    max_allowed = 1 - base_total
                    for add in not_in_set:
                        v_add = unit_vals[add]
                        if v_add < max_allowed:
                            nd = cur_deficit + v_rm - v_add
                            if nd < best_swap_def:
                                best_swap_def = nd
                                best_swap_rm = rm
                                best_swap_add = add
                if best_swap_rm is not None and best_swap_add is not None:
                    cur_set.discard(best_swap_rm)
                    cur_set.add(best_swap_add)
                    cur_total = sum(unit_vals[i] for i in cur_set)
                    cur_deficit = 1 - cur_total
                    in_set.discard(best_swap_rm)
                    in_set.add(best_swap_add)
                    not_in_set.discard(best_swap_add)
                    not_in_set.add(best_swap_rm)
                    improved = True
                    check_solution(sorted(cur_set), cur_total)
        return cur_deficit

    if best_indices:
        local_search(best_indices, max_iters=5000)
        
    import random
    random.seed(42)
    for _ in range(50):
        if time.time() >= deadline - 10:
            break
        o = list(range(n))
        random.shuffle(o)
        idxs, total = greedy(o)
        if idxs:
            check_solution(idxs, total)
            local_search(idxs, max_iters=500)
            
    if not best_indices:
        best_indices = [0]
        
    return best_indices
# EVOLVE-BLOCK-END
