<<<<<<< SEARCH
    # 5. Local Search Polish: Try to add elements back
    def local_polish(initial_set):
        active = set(initial_set)
        # Build a fast check for "can I add x?"
        # For each x, check if adding x creates any violation.
        # A violation is created if there exists a triple (x, y, z) where y, z are in active.
        
        # Precompute for each x, the pairs (y, z) such that (x, y, z) is a triple.
        # Actually, we have elem_to_triples.
        # If we add x, we must ensure no triple (x, y, z) has y, z in active.
        
        changed = True
        while changed:
            changed = False
            # Try to add any element not in active
            for x in range(1, N + 1):
                if x in active:
                    continue
                # Check if adding x is safe
                safe = True
                for t_idx in elem_to_triples[x]:
                    a, b, c = triples[t_idx]
                    # The other two elements in the triple
                    others = [y for y in (a, b, c) if y != x]
                    if others[0] in active and others[1] in active:
                        safe = False
                        break
                if safe:
                    active.add(x)
                    changed = True
        return active
=======
    # 5. Local Search Polish: Try to add elements and perform swaps
    def local_polish(initial_set):
        active = set(initial_set)
        
        changed = True
        while changed:
            changed = False
            
            # 1. Try to add any safe node
            for x in range(1, N + 1):
                if x in active:
                    continue
                safe = True
                for t_idx in elem_to_triples[x]:
                    a, b, c = triples[t_idx]
                    others = [y for y in (a, b, c) if y != x]
                    if others[0] in active and others[1] in active:
                        safe = False
                        break
                if safe:
                    active.add(x)
                    changed = True
            
            # 2. Try swaps: remove y, add x
            # Only try if x is not currently safe
            for x in range(1, N + 1):
                if x in active:
                    continue
                
                # Check if x is safe and collect conflicts if not
                safe = True
                conflicting_nodes = set()
                for t_idx in elem_to_triples[x]:
                    a, b, c = triples[t_idx]
                    others = [y for y in (a, b, c) if y != x]
                    if others[0] in active and others[1] in active:
                        safe = False
                        conflicting_nodes.add(others[0])
                        conflicting_nodes.add(others[1])
                
                if not safe:
                    # Try to remove one conflicting node to make x safe
                    # Sort conflicting nodes by degree (heuristic: remove less connected first?)
                    # For now, just iterate
                    for y in conflicting_nodes:
                        if y not in active:
                            continue
                        
                        # Temporarily remove y
                        active.remove(y)
                        
                        # Check if x is now safe
                        safe_after_removal = True
                        for t_idx in elem_to_triples[x]:
                            a, b, c = triples[t_idx]
                            others = [k for k in (a, b, c) if k != x]
                            if others[0] in active and others[1] in active:
                                safe_after_removal = False
                                break
                        
                        if safe_after_removal:
                            active.add(x)
                            changed = True
                            break # Found a valid swap for x
                        else:
                            # Restore y
                            active.add(y)
                            
        return active
>>>>>>> REPLACE

This change allows the local search to escape local optima where a node is blocking multiple additions by swapping it out.

I will also slightly increase the SA penalty to 1000 (it is already 1000) to ensure validity.
The current code has `1000`. I will keep it.

One more improvement: In `simulate_annealing`, the `apply_flips` and `revert_flips` are a bit slow due to Python loops. But they are necessary.
The main bottleneck is likely the number of SA steps.
With the improved `local_polish`, the SA's role is to find a good "basin" for the local search.
The current strategy runs SA on the top 5 greedy seeds.
Let's also run `local_polish` on the greedy seeds directly before SA, to ensure they are locally optimal.
Actually, the code does:
1. Greedy -> `greedy_results`
2. Top 5 -> SA -> `sa_set`
3. `smart_cleanup` -> `local_polish` -> `best_set`

It does NOT polish the greedy seeds before SA.
Let's add `local_polish` to the greedy results immediately.

