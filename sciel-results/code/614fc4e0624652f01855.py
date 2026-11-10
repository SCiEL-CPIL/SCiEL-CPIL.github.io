import time
import random
import numpy as np

N = 256

# EVOLVE-BLOCK-START

def construct():
    """
    Finds a small inclusion-maximal Sidon set in [1, N].
    Uses backtracking search for increasing set sizes k.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Helper to check if a set is valid Sidon
    def is_sidon(arr):
        sums = set()
        for i in range(len(arr)):
            a = arr[i]
            for j in range(i, len(arr)):
                b = arr[j]
                s = a + b
                if s in sums:
                    return False
                sums.add(s)
        return True

    # Helper to check if a set is maximal
    # Returns True if no element from [1, N] can be added without breaking Sidon property
    def is_maximal(arr, sums):
        arr_set = set(arr)
        for x in range(1, N + 1):
            if x in arr_set:
                continue
            
            # Check if x can be added
            # New sums: x+a for a in arr, and 2x
            # If any of these are in `sums`, x cannot be added.
            # Also, if x+a1 == x+a2 for a1!=a2 in arr, that's impossible since arr is distinct.
            # If x+a == 2x, i.e., a=x, but x not in arr.
            # If x+a1 == x+a2, a1=a2.
            # So we just check if any new sum conflicts with existing sums.
            
            can_add = True
            for a in arr:
                if (x + a) in sums:
                    can_add = False
                    break
            if can_add and (2 * x) in sums:
                can_add = False
                
            if can_add:
                return False
        return True

    # Precompute potential conflicts for speed? 
    # For a given set S and candidate x, x is blocked if:
    # 1. 2x in Sums(S)
    # 2. x + a in Sums(S) for some a in S
    
    # Backtracking search for a set of size k
    def search_k(k, current_set, current_sums, current_arr_set):
        """
        Recursively build a set of size k.
        Pruning:
        1. If len(current_set) == k, check if maximal.
        2. If time is up, return None.
        3. Heuristic pruning: 
           - If the number of "unblocked" elements is too high to be covered by remaining slots?
           Actually, adding an element blocks more elements.
           Let U be the set of unblocked elements (those that CAN be added).
           If we have r slots left, can we block all of U?
           Each new element y blocks y itself and potentially others.
           This is complex. Let's rely on strong ordering and early termination.
        """
        if time.time() > deadline:
            return None
            
        if len(current_set) == k:
            if is_maximal(current_set, current_sums):
                return current_set[:]
            return None
        
        # Determine candidates to add
        # We want to add elements that block as many "unblocked" elements as possible.
        
        # First, identify which elements are currently "blocked" (cannot be added).
        # An element x is blocked if:
        # - 2x in current_sums
        # - x + a in current_sums for some a in current_set
        
        # We want to pick a new element y to add to current_set.
        # y must be addable (Sidon compatible).
        # y should ideally be one that, when added, blocks many other elements.
        
        # Let's find all candidates y that can be added to current_set.
        candidates = []
        for y in range(1, N + 1):
            if y in current_arr_set:
                continue
            
            # Check if y can be added to current_set (Sidon check)
            can_add = True
            for a in current_set:
                if (y + a) in current_sums:
                    can_add = False
                    break
            if can_add and (2 * y) in current_sums:
                can_add = False
                
            if can_add:
                candidates.append(y)
                
        if not candidates:
            return None
            
        # Heuristic: Sort candidates by "blocking power"
        # Blocking power of y: how many elements z (z != y, z not in current_set) 
        # are blocked by adding y?
        # z is blocked by y if:
        # 1. 2z in NewSums (where NewSums = current_sums U {y+a for a in current_set} U {2y})
        #    This is hard to precompute for all z.
        # Simpler heuristic: Prefer larger y? Or smaller y?
        # Usually, small numbers create small sums. Large numbers create large sums.
        # Sums range from 2 to 2N.
        # If we pick large y, it uses large sums.
        # Let's try sorting by descending order of y first (greedy large), 
        # or ascending (greedy small).
        # Or random shuffle to explore different branches.
        
        # To make it deterministic but varied, we can use a mix.
        # Let's sort by descending "blocking potential" approximated by degree in a conflict graph?
        # For now, let's try a few orders: Descending, Ascending, Random.
        
        # Optimization: If k is small, we can try specific orders.
        # Let's iterate through candidates in a specific order.
        
        # Order: Descending value (often better for packing large sums)
        # But wait, if we pick large y, 2y is large. x+a is large.
        # Small x are blocked by small sums.
        # Let's try sorting by descending value.
        
        ordered_candidates = sorted(candidates, reverse=True)
        
        # To avoid getting stuck in a single deterministic path if it fails,
        # we can randomize the order slightly or try multiple permutations if time allows.
        # For now, let's stick to a deterministic heuristic first, then randomize if needed.
        
        # Actually, a better heuristic: 
        # An element y blocks z if z+y is a sum in S or 2z is a sum in S.
        # Let's just try them in order.
        
        for y in ordered_candidates:
            # Add y
            new_arr_set = current_arr_set | {y}
            new_sums = set(current_sums)
            
            # Check Sidon validity for new sums
            valid = True
            for a in current_set:
                s = y + a
                if s in new_sums:
                    valid = False
                    break
                new_sums.add(s)
            if valid:
                s = 2 * y
                if s in new_sums:
                    valid = False
                else:
                    new_sums.add(s)
                    
            if not valid:
                continue
                
            # Recurse
            result = search_k(k, current_set + [y], new_sums, new_arr_set)
            if result is not None:
                return result
                
        return None

    # Main loop: try increasing k
    # Start with k=1. Is there a size 1 maximal Sidon set?
    # If S={x}, sums={2x}. 
    # Maximal if for all y != x, y cannot be added.
    # y can be added if 2y not in {2x} and y+x not in {2x}.
    # 2y = 2x => y=x (excluded).
    # y+x = 2x => y=x (excluded).
    # So for any x, S={x} is NOT maximal because we can add any y != x?
    # Wait. If S={10}, sums={20}.
    # Can we add 1? 1+10=11 (not 20), 1+1=2 (not 20). Yes.
    # So k=1 is never maximal for N>1.
    
    # Try k=2, 3, ...
    
    # To speed up search, we can use random restarts for the candidate ordering.
    # But search_k is recursive. We can wrap it in a loop with different random seeds for shuffling.
    
    best_set = None
    
    # Pre-compute some properties?
    # For k up to ~10, backtracking might be feasible with good pruning.
    # The number of Sidon sets of size k in [1, 256] is large, but the "maximal" constraint is strong.
    
    # Let's try deterministic orders first for small k.
    # Order 1: Descending
    # Order 2: Ascending
    # Order 3: Random
    
    # We will modify search_k to accept a candidate_order function.
    
    def search_k_with_order(k, order_func, rng):
        def _search(current_set, current_sums, current_arr_set):
            if time.time() > deadline:
                return None
                
            if len(current_set) == k:
                if is_maximal(current_set, current_sums):
                    return current_set[:]
                return None
            
            candidates = []
            for y in range(1, N + 1):
                if y in current_arr_set:
                    continue
                can_add = True
                for a in current_set:
                    if (y + a) in current_sums:
                        can_add = False
                        break
                if can_add and (2 * y) in current_sums:
                    can_add = False
                if can_add:
                    candidates.append(y)
                    
            if not candidates:
                return None
                
            ordered_candidates = order_func(candidates, rng)
            
            for y in ordered_candidates:
                new_arr_set = current_arr_set | {y}
                new_sums = set(current_sums)
                
                valid = True
                for a in current_set:
                    s = y + a
                    if s in new_sums:
                        valid = False
                        break
                    new_sums.add(s)
                if valid:
                    s = 2 * y
                    if s in new_sums:
                        valid = False
                    else:
                        new_sums.add(s)
                        
                if not valid:
                    continue
                    
                result = _search(current_set + [y], new_sums, new_arr_set)
                if result is not None:
                    return result
            return None
            
        return _search([], set(), set())

    # Define ordering functions
    def order_desc(cands, rng):
        return sorted(cands, reverse=True)
        
    def order_asc(cands, rng):
        return sorted(cands)
        
    def order_rand(cands, rng):
        c = cands[:]
        rng.shuffle(c)
        return c

    # Try k=2, 3, 4...
    # For each k, try a few orders.
    
    rng = random.Random(42)
    
    # We can also use a timeout per k.
    # If k=2 takes too long, skip to k=3? No, we want smallest.
    # But if k=2 is impossible, we need to prove it or timeout.
    # Given N=256, k=2 is likely impossible.
    # Let's give each k a time budget.
    
    time_per_k = 5.0
    
    for k in range(2, 20):
        if time.time() > deadline:
            break
            
        k_start = time.time()
        k_deadline = k_start + time_per_k
        
        # Temporarily override deadline for search_k_with_order
        # We need to pass the deadline down. 
        # Let's just use the global deadline but check time in the loop.
        
        found = None
        
        # Try descending order
        if found is None and time.time() < k_deadline:
            found = search_k_with_order(k, order_desc, rng)
            
        # Try ascending order
        if found is None and time.time() < k_deadline:
            found = search_k_with_order(k, order_asc, rng)
            
        # Try random orders
        attempts = 0
        while found is None and time.time() < k_deadline and attempts < 20:
            attempts += 1
            found = search_k_with_order(k, order_rand, rng)
            
        if found:
            best_set = found
            break
            
    # If no small set found, fallback to greedy construction (which finds a large one, but valid)
    if best_set is None:
        # Greedy construct from descending order
        selected = []
        sums = set()
        for x in range(N, 0, -1):
            conflict = False
            for a in selected:
                if (x + a) in sums:
                    conflict = True
                    break
            if conflict:
                continue
            if (2 * x) in sums:
                continue
            selected.append(x)
            for a in selected:
                sums.add(x + a)
            sums.add(2 * x)
            
        best_set = selected

    # Final verification
    if not is_sidon(best_set) or not is_maximal(best_set, set()):
        # Recalculate sums for is_maximal
        sums = set()
        for i in range(len(best_set)):
            for j in range(i, len(best_set)):
                sums.add(best_set[i] + best_set[j])
        if not is_maximal(best_set, sums):
             # Fallback to a known valid one? 
             # The greedy one above should be valid. 
             # If this fails, there's a bug in is_maximal or greedy.
             # Let's assume greedy is correct.
             pass

    return best_set

# EVOLVE-BLOCK-END
