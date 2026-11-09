"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80 seconds safety margin before 90s kill
    
    def _check_divisibility(sum_set):
        """Check that no two distinct values in sum_set divide each other."""
        if len(sum_set) <= 1:
            return True
        sums = sorted(sum_set)
        n = len(sums)
        for i in range(n):
            si = sums[i]
            if si <= 0:
                continue
            # Check multiples of si
            max_val = sums[-1]
            for k in range(2, max_val // si + 1):
                if (si * k) in sum_set:
                    return False
        return True

    def _get_subset_sums(elements):
        """Get all non-empty subset sums as a set."""
        if not elements:
            return set()
        sums = set()
        n = len(elements)
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += elements[i]
            sums.add(s)
        return sums

    def _can_add_fast(current_sums, x):
        """
        Fast check: can x be added?
        Returns (True, new_sums) if valid, (False, None) otherwise.
        current_sums is a set of current non-empty subset sums.
        """
        # New sums: current_sums ∪ {x} ∪ {s+x for s in current_sums}
        # Check for duplicates first (efficiency and correctness)
        new_sums = set(current_sums)
        if x in new_sums:
            return False, None
        new_sums.add(x)
        
        added_any = False
        for s in current_sums:
            ns = s + x
            if ns in new_sums:
                return False, None
            new_sums.add(ns)
            added_any = True
        
        # Check divisibility among new_sums
        # Optimization: only need to check if any new element divides or is divided by existing ones
        # But since we're adding many elements, just do the full check on the new set if it's not too big
        # If new_sums is large, this might be slow. But for greedy, we stop early if invalid.
        
        # Efficient divisibility check
        # Sort is expensive. Let's use the multiple checking method.
        # We need to check if any a in new_sums divides any b in new_sums (a != b).
        # Since all elements are positive:
        # For each a in new_sums, check if 2a, 3a, ... are in new_sums.
        # To avoid checking every pair, we can iterate through elements.
        # However, doing this for ALL elements in new_sums every time is O(N * MaxSum/MinSum).
        
        # Alternative:
        # The new elements are {x} and {s+x for s in current_sums}.
        # Let's call these "new_vals".
        # We need to check:
        # 1. For any v in new_vals, does any u in current_sums divide v?
        # 2. For any v in new_vals, does v divide any u in current_sums? (Impossible if v > u, but v could be small? No, x>0, s>0 so v > s. So v > u is not guaranteed if x is small? Wait. s is sum of subset. s+x > s. So new sums are strictly larger than the subset sums they came from. But x itself might be smaller than some s.)
        # Actually, x could be smaller than some s.
        # Example: current sums {10, 20}. x=5. New sums: 5, 15, 25, 10, 20.
        # 5 divides 10, 15, 20, 25.
        
        # So we must check all pairs.
        # To speed up, we can just check the new elements against the whole set.
        # Let new_elements = [x] + [s+x for s in current_sums]
        # For each ne in new_elements:
        #   Check if any old_s in current_sums divides ne (ne % old_s == 0)
        #   Check if ne divides any old_s (old_s % ne == 0) -- Note: if ne > old_s, this is false.
        #   Check if ne divides any other new_element (ne2 % ne == 0)
        #   Check if any other new_element divides ne (ne % ne2 == 0)
        
        # This is roughly O(N_new * N_old + N_new^2).
        # N_new = N_old + 1.
        # So O(N_old^2).
        
        # If N_old is large (e.g. 1000), this is 10^6 operations.
        # In Python, 10^6 operations is ~0.1s.
        # If we do this 256 times (for each candidate x), that's 25.6s.
        # Plus the overhead.
        # It might be tight.
        
        # Let's try to optimize the divisibility check using the multiple method on the FULL new set,
        # but only if the set size is manageable.
        
        if len(new_sums) > 2000:
             # Too big to check quickly? 
             # Actually, if we are greedy, we might not reach this size often if we are careful.
             # But if we do, we need a fast check.
             pass
        
        # Let's stick to the multiple check on the full set, it's robust.
        # To make it faster, we can pre-sort? No, sorting is O(N log N).
        # The multiple check:
        # for a in new_sums:
        #   for k in range(2, max//a + 1):
        #     if a*k in new_sums: return False
        
        # This is fast if a is large.
        # If a is small, it's slow.
        # But small a's are rare in large sets (if we pick large numbers).
        
        # Let's implement a hybrid:
        # If len(new_sums) < 500, do the O(N^2) check with sorted list?
        # Or just the multiple check.
        
        # Let's use the multiple check.
        max_val = max(new_sums)
        for a in new_sums:
            if a <= 1:
                # 1 divides everything. If 1 is in the set and set size > 1, invalid.
                # Since we only add x>0 and sums>0, 1 is only possible if x=1.
                if len(new_sums) > 1:
                    return False, None
                else:
                    return True, new_sums
            
            # Check multiples
            # Optimization: if a > max_val // 2, no multiple can exist (except itself, which we skip)
            if a > max_val // 2:
                continue
                
            for k in range(2, max_val // a + 1):
                if (a * k) in new_sums:
                    return False, None
                    
        return True, new_sums

    def _greedy_construct(pool, deadline):
        elements = []
        current_sums = set()
        
        for x in pool:
            if time.time() > deadline:
                break
            ok, new_sums = _can_add_fast(current_sums, x)
            if ok:
                elements.append(x)
                current_sums = new_sums
                
        return elements

    best_result = [31, 32, 34]
    best_size = 3
    
    rng = random.Random(12345)
    
    # Strategy:
    # 1. Try descending order (large numbers first) - often works well.
    # 2. Try random shuffles of various pools.
    # 3. Try specific pools (primes, high range).
    
    # Pool 1: All numbers 2..256 (excluding 1 as it's problematic)
    # Actually, 1 is very restrictive. Let's exclude it from initial pools.
    
    pools = [
        list(range(256, 1, -1)),  # Descending
        list(range(2, 257)),      # Ascending
        [x for x in range(2, 257) if x % 2 != 0], # Odds
        [x for x in range(2, 257) if x % 2 == 0], # Evens
    ]
    
    # Add some random pools
    for _ in range(10):
        p = list(range(2, 257))
        rng.shuffle(p)
        pools.append(p)
        
    # High range pools
    for low in [100, 150, 200]:
        p = list(range(low, 257))
        pools.append(p)
        p2 = p[:]
        rng.shuffle(p2)
        pools.append(p2)

    for pool in pools:
        if time.time() > deadline:
            break
        result = _greedy_construct(pool, deadline)
        if len(result) > best_size:
            best_size = len(result)
            best_result = result
            
    # Local search / Extension
    # Try to extend the best result by adding numbers not in it.
    if best_size < 30 and time.time() < deadline - 5:
        current_elements = list(best_result)
        current_sums = _get_subset_sums(current_elements)
        
        # Try adding any number from 2 to 256 not in current_elements
        # Sort candidates by some heuristic? Maybe descending?
        candidates = [x for x in range(256, 1, -1) if x not in current_elements]
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for x in candidates:
                if time.time() > deadline:
                    break
                ok, new_sums = _can_add_fast(current_sums, x)
                if ok:
                    current_elements.append(x)
                    current_sums = new_sums
                    candidates.remove(x)
                    improved = True
                    if len(current_elements) > best_size:
                        best_size = len(current_elements)
                        best_result = list(current_elements)
                        # Restart candidates? No, we already removed x.
                        # But adding x might have opened up other possibilities? No, constraints are monotonic.
                        # If x couldn't be added before, it still can't.
                        # If x could be added, we added it.
                        # So we just continue.

    # Final validation
    if len(best_result) > 0:
        sums = _get_subset_sums(best_result)
        if not _check_divisibility(sums):
            # Fallback
            best_result = [31, 32, 34]
            
    return [int(x) for x in best_result]

# EVOLVE-BLOCK-END
