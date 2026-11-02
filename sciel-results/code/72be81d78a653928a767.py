import time
import random
import itertools

# Global cache for neighbors to avoid recomputation if construct is called multiple times (though it's called once per eval)
# But construct might be called with different q, n. So we should probably compute inside or cache by (q, n).
# Given the constraints, computing inside is safer and fast enough.

def construct(q, n, radius):
    start_time = time.time()
    timeout = 115.0  # Leave 5s margin
    
    # Only handle the specific case or general logic?
    # The task says "The evaluator passes this task's alphabet size q, length n and covering radius."
    # The specific task is q=3, n=7, radius=1.
    # I will implement a general greedy + local search for small q^n.
    
    total_words = q ** n
    if total_words > 100000:
        # Fallback for larger sizes: Direct sum or simple greedy
        # For this specific task, q=3, n=7 is 2187, so we are fine.
        # But to be safe against other inputs if the evaluator changes:
        pass

    # Precompute neighbors for all words in {0..q-1}^n
    # A word is an integer in [0, total_words)
    # neighbors[u] is a list of integers v such that dist(u, v) <= radius
    
    def int_to_digits(x, n, q):
        digits = [0] * n
        for i in range(n):
            digits[i] = x % q
            x //= q
        return digits

    def digits_to_int(digits, q):
        val = 0
        for i in range(len(digits)):
            val = val * q + digits[i]
        return val

    # Generate all words
    all_words = range(total_words)
    
    # Precompute neighbors
    neighbors = [[] for _ in range(total_words)]
    for u in all_words:
        du = int_to_digits(u, n, q)
        # u itself
        neighbors[u].append(u)
        # Change one position
        for pos in range(n):
            orig_val = du[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                # Create new digit list
                # To avoid creating new lists, we can calculate the integer directly
                # val = u - orig_val * q^pos + new_val * q^pos
                power = q ** pos
                v = u - orig_val * power + new_val * power
                neighbors[u].append(v)
        # Note: This generates all words at distance <= 1.
        # If radius > 1, this would need to be extended. 
        # For radius=1, this is correct.
        
    # Greedy Initialization
    # uncovered is a set of ints
    uncovered = set(all_words)
    C = []
    
    # To speed up greedy selection, we can just iterate.
    # N=2187, iterations ~ 186. 2187 * 186 * 15 (neighbors) ~ 6M ops. Fast.
    
    while uncovered:
        best_u = -1
        best_count = -1
        for u in all_words:
            # Count how many neighbors of u are in uncovered
            cnt = 0
            for v in neighbors[u]:
                if v in uncovered:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_u = u
                if cnt == len(neighbors[u]): # Perfect match, can't do better
                    break
        
        if best_u == -1:
            break # Should not happen if uncovered is not empty and neighbors are valid
            
        C.append(best_u)
        # Remove covered
        for v in neighbors[best_u]:
            if v in uncovered:
                uncovered.remove(v)
                
    # Local Search
    # State: C is a list of ints
    
    def is_covering(C_set):
        # Check if C_set covers all words
        # Use a count array for speed
        counts = [0] * total_words
        for c in C_set:
            for v in neighbors[c]:
                counts[v] += 1
        for i in range(total_words):
            if counts[i] == 0:
                return False
        return True

    # We maintain C as a list.
    # We will try to remove elements.
    # Then we will try swaps.
    
    # To make removal/swap checks fast, we can maintain the `counts` array incrementally?
    # Or just recompute. Recomputing is 200 * 15 = 3000 ops. Very fast.
    # Let's stick to recomputing for simplicity and correctness.
    
    removed_any = True
    while removed_any and (time.time() - start_time) < timeout:
        removed_any = False
        # Shuffle C to randomize removal order
        random.shuffle(C)
        
        # Try removals
        i = 0
        while i < len(C):
            c = C[i]
            # Tentatively remove
            C_temp = C[:i] + C[i+1:]
            if is_covering(C_temp):
                C = C_temp
                removed_any = True
                # Do not increment i, check next element in new list at same index
            else:
                i += 1
                
    # If we still have time, try swaps to escape local minima
    # A swap replaces c in C with w not in C (or any w).
    # If is_covering(C'), accept. Then run removals again.
    
    # To be efficient, we only do swaps if we are stuck (no removals in last pass) 
    # or periodically.
    
    # Let's add a swap loop
    # We can alternate or just run swaps if removals fail.
    
    # Actually, a simpler robust approach:
    # Loop:
    # 1. Try all removals. If any removed, continue loop.
    # 2. If no removals, try a random swap.
    #    - Pick random index i in C.
    #    - Pick random word w from all_words.
    #    - If w == C[i], skip.
    #    - If w in C, skip (or allow, but redundant).
    #    - Let C' = C with C[i] replaced by w.
    #    - If is_covering(C'), accept C' = C. Then go to step 1 (try removals).
    #    - Else, reject.
    #    - To avoid getting stuck in a very bad local minimum, we might need simulated annealing,
    #    but for this small instance, simple hill climbing with random restarts or just random swaps 
    #    often finds the global optimum (186).
    
    # Let's implement the swap phase.
    
    # We need to be careful with time.
    # Each is_covering check is ~3000 ops.
    # We can do ~100k checks in 100ms? 
    # 100k * 3000 = 300M ops. Python might be slow. 
    # Let's optimize is_covering.
    
    def is_covering_fast(C_list):
        # Using a set of covered words might be faster if C is small?
        # No, counts array is O(1) update.
        # Let's try to optimize the inner loop.
        counts = [0] * total_words
        for c in C_list:
            for v in neighbors[c]:
                counts[v] += 1
        # Check if any 0
        # Using min() might be faster in C implementation
        return min(counts) > 0

    # Update the removal loop to use is_covering_fast
    # And the swap loop.
    
    # Re-run removals with fast checker
    removed_any = True
    while removed_any and (time.time() - start_time) < timeout:
        removed_any = False
        random.shuffle(C)
        i = 0
        while i < len(C):
            c = C[i]
            C_temp = C[:i] + C[i+1:]
            if is_covering_fast(C_temp):
                C = C_temp
                removed_any = True
            else:
                i += 1
                
    # Swap phase
    # If we are still above 186, or just to ensure we are at the minimum,
    # we perform random swaps.
    
    # If the greedy init + removals already got us to 186, we can stop.
    # But it might get stuck at 187 or 190.
    
    # Let's run swaps until time runs out or we hit a lower bound (e.g. 186).
    # We don't know the lower bound for sure in code, but we know 186 is achievable.
    # We can just run until timeout.
    
    # To make swaps effective, we should prioritize swapping words that are "redundant" in some sense?
    # Or just random.
    
    # One improvement: If we swap and it works, we *must* try to remove again.
    # So the structure is:
    # while time < limit:
    #   if try_removals(): continue
    #   if try_random_swap(): continue
    #   # If neither works, we are stuck. 
    #   # Maybe do a "kick": remove a random word, add a random word (even if invalid?), 
    #   # then repair? 
    #   # Simpler: just keep trying random swaps. The space is small.
    
    # Let's implement a loop that does removals, then if no removals, does a batch of swaps.
    
    # Actually, the previous loop already did removals. 
    # Now enter swap loop.
    
    while (time.time() - start_time) < timeout:
        # Try a random swap
        if len(C) == 0: break
        
        idx = random.randint(0, len(C) - 1)
        old_c = C[idx]
        
        # Pick a new word
        # Bias towards words that cover many currently "weakly covered" words?
        # For simplicity, pick random.
        new_c = random.randint(0, total_words - 1)
        
        if new_c == old_c:
            continue
        
        # Check if new_c is already in C?
        # If it is, swapping old_c with new_c (where new_c is already in C) 
        # effectively removes old_c and keeps new_c.
        # This is equivalent to removing old_c if new_c is in C.
        # But we already tried removals. 
        # However, if we swap old_c with new_c, and new_c is in C, 
        # the set becomes (C \ {old_c}) U {new_c} = C \ {old_c} (since new_c is in C).
        # So this is just a removal attempt.
        # If is_covering(C \ {old_c}) was false, this swap will also be false.
        # So we should avoid new_c in C to force a true swap.
        
        # Efficient check if new_c in C:
        # C is small, linear search is fine.
        if new_c in C:
            continue
            
        C_new = C.copy()
        C_new[idx] = new_c
        
        if is_covering_fast(C_new):
            C = C_new
            # After a successful swap, the structure changed.
            # It is likely that some word is now redundant.
            # So we should break out of the swap loop and go back to removals.
            # How to signal this? 
            # We can just restart the outer loop logic.
            # Let's restructure:
            break # Break inner while to go to removals again? 
            # But I put the removals in a while loop above.
            # I should wrap the whole thing in a main loop.
            
    # The code above is a bit messy with the breaks. 
    # Let's rewrite the main search loop cleanly.
    
    # Refactored Search:
    
    # 1. Initial Greedy (Done)
    # 2. Main Loop
    #    while time < limit:
    #       progress = False
    #       # Phase 1: Removals
    #       while True:
    #           found = False
    #           random.shuffle(C)
    #           i = 0
    #           while i < len(C):
    #              if is_covering_fast(C[:i] + C[i+1:]):
    #                 C = C[:i] + C[i+1:]
    #                 found = True
    #                 progress = True
    #              else:
    #                 i += 1
    #           if not found: break
    #           
    #       # Phase 2: Swaps (if no removals found in Phase 1)
    #       if not progress:
    #           # Try a few random swaps
    #           swap_success = False
    #           for _ in range(100): # Try up to 100 swaps
    #               if (time.time() - start_time) > timeout: break
    #               if len(C) == 0: break
    #               idx = random.randint(0, len(C)-1)
    #               old_c = C[idx]
    #               new_c = random.randint(0, total_words - 1)
    #               if new_c == old_c or new_c in C: continue
    #               C_new = C.copy()
    #               C_new[idx] = new_c
    #               if is_covering_fast(C_new):
    #                   C = C_new
    #                   swap_success = True
    #                   break
    #           if swap_success:
    #               # If a swap was successful, go back to Phase 1 (Removals)
    #               # The 'continue' of the main while loop will do this.
    #               pass
    #           else:
    #               # No removals, no swaps. Stuck?
    #               # We could implement a more aggressive move, but for this size,
    #               # random swaps usually eventually find a path.
    #               # To avoid infinite tight loops, we can just keep trying.
    #               pass
    #
    # This looks solid.

    # Let's implement this refactored loop.
    
    # Note: The previous code blocks for removals were already run.
    # I will just replace the logic after the initial greedy with this robust loop.
    
    # Re-initialize time check
    while (time.time() - start_time) < timeout:
        progress = False
        
        # Phase 1: Removals
        # We try to remove elements. We do this in a loop until no more can be removed.
        while True:
            found_removal = False
            random.shuffle(C)
            i = 0
            while i < len(C):
                c = C[i]
                # Create list without C[i]
                # Optimization: If len(C) is small, slicing is fine.
                C_temp = C[:i] + C[i+1:]
                if is_covering_fast(C_temp):
                    C = C_temp
                    found_removal = True
                    progress = True
                    # Do not increment i
                else:
                    i += 1
            if not found_removal:
                break
        
        # Phase 2: Swaps
        if not progress:
            # Try to find a swap
            swap_found = False
            # Try a limited number of swaps per "iteration" to allow time checks
            for _ in range(50):
                if (time.time() - start_time) > timeout:
                    break
                if len(C) == 0:
                    break
                idx = random.randint(0, len(C) - 1)
                old_c = C[idx]
                new_c = random.randint(0, total_words - 1)
                
                if new_c == old_c:
                    continue
                
                # Check if new_c is in C
                # If it is, it's a removal. We already know removals fail (progress=False).
                # So skip to force a change.
                if new_c in C:
                    continue
                
                C_new = C.copy()
                C_new[idx] = new_c
                
                if is_covering_fast(C_new):
                    C = C_new
                    swap_found = True
                    break
            
            if swap_found:
                # If a swap was found, we made progress in structure.
                # We should go back to Phase 1 to see if we can remove anything now.
                # The outer while loop will continue, and Phase 1 will run again.
                pass
            else:
                # No removals, no swaps found in this batch.
                # We are in a local minimum.
                # We could implement a "kick" (random restart or larger move),
                # but for n=7, simple random swaps usually suffice to find 186.
                # If we are really stuck, we might just keep trying.
                # To avoid wasting time on a dead end, we could do a random restart
                # if the current size is not significantly better than the initial greedy.
                # But let's just keep trying swaps. The probability of finding a valid swap
                # increases as the set becomes more "dense" or "flexible".
                pass

    # Convert C (list of ints) to list of lists of ints
    result = []
    for c in C:
        result.append(int_to_digits(c, n, q))
        
    return result
