"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    # Prescribed elements
    prescribed = [1, 2, 5, 11]
    
    # Bitmask for sums: sum s is represented as bit (s-2) since min sum is 1+1=2, max is 284+284=568
    # Actually sums range from 1+2=3 (distinct) but we use a<=b so min is 1+1=2
    # Let's use bit position (s - 2) for sum s, so we need bits 0..566 (567 bits)
    # We'll use Python integers as bitmasks
    
    def compute_sums_bitmask(arr):
        """Compute bitmask of all pairwise sums a+b where a<=b from sorted arr."""
        mask = 0
        n = len(arr)
        for i in range(n):
            a = arr[i]
            for j in range(i, n):
                b = arr[j]
                s = a + b
                mask |= (1 << (s - 2))
        return mask

    def can_add_bitmask(candidate, arr, mask):
        """Check if candidate can be added to arr without creating duplicate sums."""
        # Need to check candidate + y for all y in arr (y <= candidate is fine since we add all)
        # Actually we need candidate + y for all y in arr, plus 2*candidate
        # But we must ensure these sums don't already exist in mask
        # And also that candidate + y doesn't conflict with any existing sum
        for y in arr:
            s = candidate + y
            if (mask >> (s - 2)) & 1:
                return False
        s = 2 * candidate
        if (mask >> (s - 2)) & 1:
            return False
        return True

    def add_to_bitmask(candidate, arr, mask):
        """Add candidate to arr and update mask. Returns new mask."""
        new_mask = mask
        for y in arr:
            s = candidate + y
            new_mask |= (1 << (s - 2))
        s = 2 * candidate
        new_mask |= (1 << (s - 2))
        return new_mask

    # Initial set with prescribed elements
    base_arr = sorted(prescribed)
    base_mask = compute_sums_bitmask(base_arr)
    
    best_set = base_arr[:]
    best_mask = base_mask
    best_size = len(best_set)

    def greedy_fill(arr, mask, candidates, time_limit):
        """Greedily add elements from candidates to arr/mask."""
        local_arr = arr[:]
        local_mask = mask
        for x in candidates:
            if time.time() > time_limit:
                break
            if x in local_arr:
                continue
            if can_add_bitmask(x, local_arr, local_mask):
                local_arr.append(x)
                local_mask = add_to_bitmask(x, local_arr[:-1], local_mask)
        return local_arr, local_mask

    # Initial greedy fill
    candidates = list(range(12, 285))
    greedy_arr, greedy_mask = greedy_fill(base_arr, base_mask, candidates, deadline)
    
    if len(greedy_arr) > best_size:
        best_size = len(greedy_arr)
        best_set = greedy_arr[:]
        best_mask = greedy_mask

    # Local search: try to improve by removing one element and re-adding
    # Use randomized restarts with different orderings
    rng = random.Random(42)
    
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        # Start from current best or a perturbed version
        if iteration % 10 == 0:
            # Perturb: remove a random non-prescribed element
            removable = [x for x in best_set if x not in prescribed]
            if not removable:
                continue
            remove_idx = removable.index(rng.choice(removable))
            start_set = best_set[:remove_idx] + best_set[remove_idx+1:]
            start_mask = compute_sums_bitmask(start_set)
        else:
            start_set = best_set[:]
            start_mask = best_mask
        
        # Shuffle candidates for diversity
        shuffled_candidates = list(range(1, 285))
        rng.shuffle(shuffled_candidates)
        
        # Greedy fill from perturbed start
        new_arr, new_mask = greedy_fill(start_set, start_mask, shuffled_candidates, deadline)
        
        if len(new_arr) > best_size:
            best_size = len(new_arr)
            best_set = new_arr[:]
            best_mask = new_mask

    # Final local search: try swapping elements
    for iteration in range(300):
        if time.time() > deadline:
            break
        
        # Try to improve by removing one and trying to add a better one
        removable = [x for x in best_set if x not in prescribed]
        if not removable:
            break
        
        improve_found = False
        for remove_val in removable:
            if time.time() > deadline:
                break
            
            remove_idx = best_set.index(remove_val)
            test_set = best_set[:remove_idx] + best_set[remove_idx+1:]
            test_mask = compute_sums_bitmask(test_set)
            
            # Try to add candidates in random order
            candidates = [x for x in range(1, 285) if x not in test_set]
            rng.shuffle(candidates)
            
            new_arr, new_mask = greedy_fill(test_set, test_mask, candidates, deadline)
            
            if len(new_arr) > best_size:
                best_size = len(new_arr)
                best_set = new_arr[:]
                best_mask = new_mask
                improve_found = True
                break
            elif len(new_arr) == best_size:
                # Keep if different (for diversity)
                if new_arr != best_set:
                    # Prefer sets with more room for future additions
                    # (heuristic: smaller max element might be better, but let's just keep first)
                    pass

    return best_set
# EVOLVE-BLOCK-END
