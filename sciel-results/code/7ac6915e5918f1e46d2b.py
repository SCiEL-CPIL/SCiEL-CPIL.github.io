import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # Pre-compute factorization vectors for numbers 2..40
    # Primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    
    vectors = {}
    for n in range(2, 41):
        exponents = [0] * num_primes
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                exponents[i] += 1
                temp //= p
        vectors[n] = tuple(exponents)

    # Fast check for sum-distinctness
    # We maintain the set of all subset sums of the current selection.
    # If we add a new vector v, the new set of sums is S U (S + v).
    # If (S + v) intersects S, then there is a collision.
    
    def check_addition(current_sums, new_vec):
        """
        Returns True if adding new_vec to a set with subset sums 'current_sums' 
        maintains distinctness.
        Also returns the new set of sums if valid, else None.
        """
        # We need to check if any sum in (current_sums + new_vec) is already in current_sums.
        # Equivalent to: exists s1, s2 in current_sums such that s1 + new_vec == s2?
        # Or simply: does (current_sums + new_vec) intersect current_sums?
        
        new_sums = set()
        # Optimization: We can just generate the shifted sums and check intersection
        for s in current_sums:
            new_s = tuple(s[i] + new_vec[i] for i in range(num_primes))
            if new_s in current_sums:
                return False, None
            new_sums.add(new_s)
        
        # If no intersection, the new set of sums is current_sums U new_sums
        # But we don't need to return the full set for every step in B&B if we backtrack properly,
        # but for B&B we usually need to carry state.
        return True, current_sums.union(new_sums)

    # Branch and Bound
    # We sort candidates. To maximize the chance of finding large sets early,
    # we might sort by some heuristic (e.g., number of prime factors, or just natural order).
    # Let's stick to natural order 2..40 for determinism and simplicity, 
    # but we can reorder if needed. Actually, ordering by "rarity" of prime factors might help pruning?
    # Let's try natural order first.
    
    candidates = list(range(2, 41))
    
    best_set = []
    best_size = 0
    
    # State: (current_index, current_set, current_sums)
    # We use recursion.
    
    # To speed up, we can pre-calculate the "rank" or potential contribution of remaining items?
    # A simple bound: The maximum number of items we can add is len(candidates) - current_index.
    # So if len(current_set) + (len(candidates) - current_index) <= best_size, prune.
    
    # A tighter bound:
    # The vectors are in R^12. The maximum size of a sum-distinct set is limited by the geometry.
    # However, a simple counting bound is often sufficient if the best solution is found early.
    
    # Let's implement the B&B.
    
    def branch_and_bound(idx, current_set, current_sums):
        nonlocal best_set, best_size
        
        # Pruning 1: Time check
        if time.time() > deadline:
            return
            
        # Pruning 2: Upper bound on size
        # Even if we add all remaining candidates, can we beat the best?
        max_possible = len(current_set) + (len(candidates) - idx)
        if max_possible <= best_size:
            return

        # Base case: no more candidates
        if idx == len(candidates):
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
            return

        # Option 1: Skip candidates[idx]
        branch_and_bound(idx + 1, current_set, current_sums)
        
        # Option 2: Include candidates[idx]
        # Check if adding it is valid
        vec = vectors[candidates[idx]]
        valid, new_sums = check_addition(current_sums, vec)
        
        if valid:
            # If we are already better than best, we might want to update best early?
            # But we need the full set.
            new_set = current_set + [candidates[idx]]
            branch_and_bound(idx + 1, new_set, new_sums)

    # Initial state
    initial_sums = {tuple([0]*num_primes)}
    
    # To help find a good lower bound quickly, let's run a greedy first.
    # Greedy: Add numbers in order if possible.
    greedy_set = []
    greedy_sums = initial_sums
    for n in candidates:
        vec = vectors[n]
        valid, new_sums = check_addition(greedy_sums, vec)
        if valid:
            greedy_set.append(n)
            greedy_sums = new_sums
            
    if len(greedy_set) > best_size:
        best_size = len(greedy_set)
        best_set = greedy_set

    # Now run B&B
    # We can pass the greedy solution as the initial best to help pruning.
    branch_and_bound(0, [], initial_sums)
    
    # Verify result
    # The B&B should guarantee validity by construction (check_addition), 
    # but let's double check if time allows.
    if time.time() < deadline - 2:
        # Re-verify best_set
        # Re-calculate sums for best_set to be 100% sure
        final_sums = {tuple([0]*num_primes)}
        is_valid = True
        for n in best_set:
            vec = vectors[n]
            valid, _ = check_addition(final_sums, vec)
            if not valid:
                is_valid = False
                break
            # Update final_sums for next iteration
            # check_addition doesn't return the new set if we only care about validity in this loop
            # Let's just re-run the full check for the final set
            pass
        
        # Simpler final check
        def is_sum_distinct(nums):
            k = len(nums)
            if k == 0: return True
            vecs = [vectors[x] for x in nums]
            seen = set()
            seen.add(tuple([0]*num_primes))
            sums = [None] * (1 << k)
            sums[0] = tuple([0]*num_primes)
            for mask in range(1, 1 << k):
                lsb = mask & (-mask)
                idx = lsb.bit_length() - 1
                prev_mask = mask ^ lsb
                prev_sum = sums[prev_mask]
                v = vecs[idx]
                new_sum = tuple(prev_sum[i] + v[i] for i in range(num_primes))
                if new_sum in seen:
                    return False
                seen.add(new_sum)
                sums[mask] = new_sum
            return True

        if not is_sum_distinct(best_set):
            # Fallback to greedy if B&B somehow produced invalid (shouldn't happen)
            best_set = greedy_set
            if not is_sum_distinct(best_set):
                best_set = [2]

    return [int(x) for x in best_set]
