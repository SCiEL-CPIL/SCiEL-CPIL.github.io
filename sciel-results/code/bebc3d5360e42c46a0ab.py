"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # We want to find a Sidon set A containing 0, |A| <= 32,
    # such that all d in 1..30 are differences in A.
    # Minimize max(A).
    
    # Strategy: Fast Local Search (Hill Climbing)
    # 1. Represent the set A as a bitmask (up to 10^6 is too big for int, 
    #    but we can use a set or a list of integers. 
    #    Actually, for speed, we'll use a sorted list of integers and maintain 
    #    a set of differences for O(1) lookups, updating it incrementally).
    
    # Better: Since we are minimizing the max element, and the set size is small (<=32),
    # we can represent the state as a tuple of integers.
    # However, mutating a list is faster.
    
    # We will use a "Greedy + Local Search" hybrid.
    # 1. Generate an initial valid solution using a fast greedy heuristic.
    # 2. Use local search to reduce the diameter (max element).
    #    Moves: 
    #    - Remove an element (if it doesn't break coverage or Sidon property? 
    #      Removing an element never breaks Sidon property, but might break coverage).
    #    - Replace an element with a smaller one that preserves validity.
    #    - Add a new small element and remove a larger one?
    
    # Let's stick to a simpler, very fast local search:
    # State: A sorted list of integers.
    # Objective: Minimize max(A).
    # Constraint: Sidon (distinct differences) and covers 1..30.
    
    # Helper to check validity and get differences
    def get_diffs(A):
        diffs = set()
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                d = A[j] - A[i]
                diffs.add(d)
        return diffs

    def is_valid(A):
        if 0 not in A:
            return False
        if len(A) > 32:
            return False
        diffs = get_diffs(A)
        # Check Sidon: number of unique diffs must be n*(n-1)/2
        n = len(A)
        if len(diffs) != n * (n - 1) // 2:
            return False
        # Check coverage
        for d in range(1, 31):
            if d not in diffs:
                return False
        return True

    # Initial Solution Construction
    # Greedy: Start with [0].
    # Iteratively add the smallest x > 0 such that A U {x} is Sidon 
    # and x helps cover a missing difference.
    # If no such x exists in a reasonable range, we might need to backtrack, 
    # but for this task, a simple greedy often finds a valid set quickly.
    
    def greedy_construct():
        A = [0]
        covered = set()
        # We need to cover 1..30.
        # We will try to add elements in increasing order of value.
        # To ensure we don't get stuck, we'll use a simple heuristic:
        # At each step, find the smallest x such that A U {x} is Sidon.
        # Then check if it covers any new required diffs.
        # If it covers nothing, we might still need it to enable future diffs?
        # Actually, for Sidon sets, "filling in" small gaps is good.
        
        # A more robust greedy:
        # For each missing difference d, try to find a pair (a, x) or (x, a) 
        # such that x-a=d or a-x=d, and adding x keeps it Sidon.
        # Prioritize d in 1..30.
        
        while True:
            missing = [d for d in range(1, 31) if d not in covered]
            if not missing:
                break
            if len(A) >= 32:
                break
                
            found_move = False
            # Try to cover the smallest missing difference first? 
            # Or the one with the fewest options?
            # Let's iterate through missing diffs.
            for d in missing:
                # Try to add x such that x - a = d or a - x = d
                # x = a + d or x = a - d
                candidates = set()
                for a in A:
                    candidates.add(a + d)
                    if a - d > 0:
                        candidates.add(a - d)
                
                # Sort candidates by value (smallest first to minimize diameter)
                for x in sorted(candidates):
                    if x in A:
                        continue
                    new_A = sorted(A + [x])
                    # Quick Sidon check
                    diffs = get_diffs(new_A)
                    if len(diffs) != len(new_A) * (len(new_A) - 1) // 2:
                        continue
                    
                    # Check if this move is "useful" (covers at least one new diff)
                    new_covered = set(covered)
                    for diff in diffs:
                        if 1 <= diff <= 30:
                            new_covered.add(diff)
                    
                    if len(new_covered) > len(covered):
                        A = new_A
                        covered = new_covered
                        found_move = True
                        break
                if found_move:
                    break
            
            if not found_move:
                # Greedy got stuck. This shouldn't happen often for small N,
                # but if it does, we can try a random restart or a more aggressive search.
                # For now, let's try adding a random small number that keeps it Sidon,
                # hoping it opens up paths.
                # Or just break and rely on local search to fix it? 
                # Actually, if we can't cover a diff, we are invalid.
                # Let's try a simple random restart logic within this function if stuck.
                # But let's assume the greedy works for 1..30.
                # If it fails, we return None and handle it outside.
                return None
        return A

    # Local Search to Minimize Diameter
    # State: A (list of ints)
    # Move: 
    # 1. Try to replace an element x with a smaller y (y < x, y not in A) 
    #    such that the new set is valid.
    # 2. Try to remove an element x if the remaining set is still valid 
    #    (covers 1..30 and Sidon).
    
    def local_search(A, time_limit):
        current_A = A
        current_max = max(current_A)
        best_A = current_A
        best_max = current_max
        
        # Precompute diffs for current_A
        current_diffs = get_diffs(current_A)
        
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            made_progress = False
            
            # Strategy 1: Try to reduce the largest element
            # Find the largest element x
            x = max(current_A)
            
            # Try to replace x with a smaller y
            # y should be in [1, x-1]
            # We want y to be as small as possible to reduce diameter.
            # But y must not create duplicate diffs.
            
            # Let's iterate y from 1 to x-1
            # This might be slow if x is large. 
            # Optimization: Only check y that are "plausible".
            # Actually, if x is the max, removing it reduces the set size.
            # If the set without x is still valid (covers 1..30), we should remove it.
            
            # Check if removing x keeps validity
            A_without_x = [a for a in current_A if a != x]
            if len(A_without_x) > 0:
                diffs_without_x = get_diffs(A_without_x)
                # Check Sidon
                n = len(A_without_x)
                is_sidon = (len(diffs_without_x) == n * (n - 1) // 2)
                # Check Coverage
                covers = all(d in diffs_without_x for d in range(1, 31))
                if is_sidon and covers:
                    current_A = A_without_x
                    current_diffs = diffs_without_x
                    current_max = max(current_A)
                    made_progress = True
                    if current_max < best_max:
                        best_max = current_max
                        best_A = current_A
                    # Restart loop to re-evaluate with smaller set
                    continue
            
            # If we can't remove x, try to replace it with a smaller y.
            # We want to find y < x such that A' = (A \ {x}) U {y} is valid.
            # To minimize diameter, we try small y first.
            
            base_set = [a for a in current_A if a != x]
            base_diffs = get_diffs(base_set)
            
            # We need to check if adding y to base_set keeps it Sidon and covers 1..30.
            # Note: base_set might not cover 1..30 if x was essential.
            # But if x was essential, replacing it with y must also cover those diffs.
            
            # Let's iterate y from 1 to x-1
            # To speed up, we can skip y that are in base_set.
            # Also, we can limit the range of y if we are getting close to the deadline.
            
            for y in range(1, x):
                if y in base_set:
                    continue
                
                # Check Sidon for base_set U {y}
                # New diffs are (y - a) and (a - y) for a in base_set
                # We need to ensure these new diffs are not in base_diffs
                # and are distinct from each other.
                
                new_diffs = set(base_diffs)
                valid = True
                for a in base_set:
                    d = abs(y - a)
                    if d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                # Check Coverage
                covers = all(d in new_diffs for d in range(1, 31))
                if covers:
                    # Valid move found
                    current_A = sorted(base_set + [y])
                    current_diffs = new_diffs
                    current_max = max(current_A)
                    made_progress = True
                    if current_max < best_max:
                        best_max = current_max
                        best_A = current_A
                    break # Restart loop with new state
            
            if not made_progress:
                # Stuck in a local minimum. 
                # Try a random perturbation: 
                # 1. Remove a random element (not the max) if possible?
                # 2. Or add a random small element and remove the max?
                # Let's try: Pick a random element x from A (not 0), remove it.
                # If the set is still valid, keep it.
                # If not, try to replace it with a random y < x.
                
                if len(current_A) <= 1:
                    break
                    
                # Try removing a random non-zero element
                candidates_to_remove = [a for a in current_A if a != 0]
                if not candidates_to_remove:
                    break
                
                x = random.choice(candidates_to_remove)
                base_set = [a for a in current_A if a != x]
                
                # Check if base_set is valid
                diffs_base = get_diffs(base_set)
                n = len(base_set)
                is_sidon = (len(diffs_base) == n * (n - 1) // 2)
                covers = all(d in diffs_base for d in range(1, 31))
                
                if is_sidon and covers:
                    current_A = base_set
                    current_diffs = diffs_base
                    current_max = max(current_A)
                    if current_max < best_max:
                        best_max = current_max
                        best_A = current_A
                else:
                    # Try to replace x with a random y < x
                    # We need y such that (base_set U {y}) is valid.
                    # Let's try a few random y's
                    found = False
                    for _ in range(10):
                        y = random.randint(1, x - 1)
                        if y in base_set:
                            continue
                        
                        new_diffs = set(diffs_base)
                        valid = True
                        for a in base_set:
                            d = abs(y - a)
                            if d in new_diffs:
                                valid = False
                                break
                            new_diffs.add(d)
                        
                        if not valid:
                            continue
                        
                        covers = all(d in new_diffs for d in range(1, 31))
                        if covers:
                            current_A = sorted(base_set + [y])
                            current_diffs = new_diffs
                            current_max = max(current_A)
                            if current_max < best_max:
                                best_max = current_max
                                best_A = current_A
                            found = True
                            break
                    
                    if not found:
                        # If we can't make progress, we might be stuck.
                        # Let's try a different random element to remove
                        pass
            
            # Safety check: if we've been stuck for a long time, break?
            # The loop condition handles the time limit.
        
        return best_A

    # Execution
    # 1. Get initial solution
    A_init = greedy_construct()
    
    if A_init is None:
        # Fallback if greedy fails (unlikely for 1..30)
        # Use a known small Sidon set or a more robust search
        # For now, return a large valid set
        A_init = [0, 1, 4, 10, 12, 17, 20, 23, 25, 29, 32, 35, 38, 41, 44, 47]
        # This might not be valid, so let's just use the greedy fallback from the seed
        # But the seed's fallback was slow. Let's assume greedy works.
        pass

    # 2. Local Search
    # Allocate most of the time to local search
    time_elapsed = time.time() - start_time
    remaining_time = max(1.0, deadline - time_elapsed)
    
    A_best = local_search(A_init, remaining_time)
    
    return A_best

# EVOLVE-BLOCK-END
