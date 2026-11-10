def construct():
    from math import gcd
    import random

    n = 128
    # Precompute gcd
    gcd_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            gcd_table[i][j] = gcd(i, j)

    # Identify all bad triples
    # A triple (a, b, c) is bad if gcd(a,b) == gcd(a,c) == gcd(b,c)
    # We store for each element x, the set of pairs (u, v) that form a bad triple with x
    # u < v to avoid duplicates in pair representation
    
    bad_pairs_for = [set() for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            for c in range(b + 1, n + 1):
                g_ab = gcd_table[a][b]
                g_ac = gcd_table[a][c]
                g_bc = gcd_table[b][c]
                if g_ab == g_ac == g_bc:
                    # Triple (a, b, c) is bad
                    # For a, the conflicting pair is (b, c)
                    bad_pairs_for[a].add((b, c))
                    # For b, the conflicting pair is (a, c)
                    bad_pairs_for[b].add((a, c))
                    # For c, the conflicting pair is (a, b)
                    bad_pairs_for[c].add((a, b))

    # Helper to check if adding x to current set is valid
    # in_set: list of booleans
    def can_add(x, in_set):
        for u, v in bad_pairs_for[x]:
            if in_set[u] and in_set[v]:
                return False
        return True

    # Greedy construction
    def greedy_construct(order):
        in_set = [False] * (n + 1)
        current_set = set()
        for x in order:
            if can_add(x, in_set):
                in_set[x] = True
                current_set.add(x)
        return current_set, in_set

    # Local Search: Try to improve by swapping
    # We try to find a swap (remove u, add v) that keeps validity and increases size (impossible for swap) 
    # OR we try to just add more elements? 
    # Actually, since we start with a maximal set (greedy), we can't just add.
    # We must perform swaps to potentially allow more elements later? 
    # Or we can do a "kick" strategy: remove k elements, add as many as possible.
    
    # Let's use a simple hill climbing with swaps.
    # If we remove u and add v, size stays same.
    # To increase size, we need to remove u and add v, w... 
    # Let's try: Remove u. Then try to add all possible elements not in set.
    
    def improve(current_set, in_set):
        best_set = current_set
        best_in_set = in_set[:]
        best_size = len(current_set)
        
        # Try removing each element u
        for u in list(current_set):
            # Temporarily remove u
            in_set[u] = False
            temp_set = current_set - {u}
            
            # Try to add any element not in temp_set
            # We should try adding elements that have low conflict with current set
            # To be fast, just iterate through all not-in-set elements
            added_any = False
            for v in range(1, n + 1):
                if not in_set[v]:
                    if can_add(v, in_set):
                        in_set[v] = True
                        temp_set.add(v)
                        added_any = True
            
            # If we added more than we removed (which is always >= 1 if added_any), 
            # the size increased or stayed same.
            # If size increased, keep it.
            if len(temp_set) > best_size:
                best_set = temp_set
                best_in_set = in_set[:]
                best_size = len(temp_set)
            else:
                # Restore u
                in_set[u] = True
                # Remove any added elements to restore original state?
                # It's easier to just reset in_set from best_in_set if we don't keep it?
                # No, we modified in_set. We need to be careful.
                # Let's reconstruct in_set from temp_set? No, that's slow.
                # Let's just track what was added and remove them.
                # Actually, simpler: just use a copy of in_set for the trial?
                # Copying a list of 128 bools is cheap.
                pass

        return best_set, best_in_set

    # The above improve function is buggy because it modifies in_set in place and doesn't restore properly if not kept.
    # Let's rewrite improve to be safe.
    
    def improve_safe(current_set, in_set):
        best_set = current_set
        best_in_set = in_set[:]
        best_size = len(current_set)
        
        for u in list(current_set):
            trial_in_set = in_set[:]
            trial_set = set(current_set)
            
            # Remove u
            trial_in_set[u] = False
            trial_set.remove(u)
            
            # Try to add all possible
            for v in range(1, n + 1):
                if not trial_in_set[v]:
                    if can_add(v, trial_in_set):
                        trial_in_set[v] = True
                        trial_set.add(v)
                        
            if len(trial_set) > best_size:
                best_set = trial_set
                best_in_set = trial_in_set
                best_size = len(trial_set)
                
        return best_set, best_in_set

    # Main loop with randomization
    random.seed(42)
    best_overall = set()
    best_overall_size = 0
    
    # Run multiple greedy passes with different orders
    for _ in range(50):
        order = list(range(1, n + 1))
        random.shuffle(order)
        current_set, in_set = greedy_construct(order)
        
        if len(current_set) > best_overall_size:
            best_overall = current_set
            best_overall_size = len(current_set)
            
        # Local search improvement
        for _ in range(100): # 100 iterations of local search
            new_set, new_in_set = improve_safe(current_set, in_set)
            if len(new_set) > len(current_set):
                current_set = new_set
                in_set = new_in_set
                if len(current_set) > best_overall_size:
                    best_overall = current_set
                    best_overall_size = len(current_set)
            else:
                break # No improvement, stop local search for this start

    return list(best_overall)
