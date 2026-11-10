# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import sys

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square mod 205.
    Uses greedy initialization and local search with swap-and-add moves and random restarts.
    """
    m = 205
    k = 2
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 580  # Leave 20s buffer for safety

    # Compute forbidden differences (nonzero squares mod 205)
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, k, m))
    forbidden.discard(0)
    
    # Precompute conflict lists
    # neighbors[i] contains all j such that (i-j) % m is in forbidden
    neighbors = [[] for _ in range(m)]
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            d = (i - j) % m
            if d in forbidden:
                neighbors[i].append(j)
                
    # Helper to check if a candidate can be added to a set
    # Returns True if candidate has no conflicts with any element in current_set
    def can_add(candidate, current_set):
        for s in current_set:
            if (candidate - s) % m in forbidden:
                return False
        return True

    # Greedy initialization with random order
    def greedy_init():
        selected = []
        order = list(range(m))
        random.shuffle(order)
        for candidate in order:
            ok = True
            for s in selected:
                if (candidate - s) % m in forbidden:
                    ok = False
                    break
            if ok:
                selected.append(candidate)
        return selected

    # Local search
    def local_search(initial_set, time_limit):
        current_set = set(initial_set)
        current_size = len(current_set)
        best_set = list(current_set)
        best_size = current_size
        
        no_improve_count = 0
        max_no_improve = 50 # Restart local search phase if stuck
        
        while time.time() < time_limit:
            improved = False
            
            # Phase 1: Try to add any element that has no conflicts
            # We iterate through all elements not in the set
            not_in_set = [i for i in range(m) if i not in current_set]
            # Shuffle to avoid bias
            random.shuffle(not_in_set)
            
            for candidate in not_in_set:
                if can_add(candidate, current_set):
                    current_set.add(candidate)
                    current_size += 1
                    improved = True
                    # If we added one, we should re-check because adding one might
                    # not block others, but it might block some. 
                    # Actually, adding one element only adds constraints, it doesn't remove them.
                    # So if an element was not addable before, it won't become addable later
                    # unless we remove something. 
                    # So we can just continue checking the rest of not_in_set.
                    # However, the list `not_in_set` is static. 
                    # If we add `candidate`, we don't need to re-evaluate `candidate` obviously.
                    # We also don't need to re-evaluate elements that were already rejected 
                    # because they conflicted with elements still in the set.
                    # But elements rejected because they conflicted with an element 
                    # that we *might* swap out later are not relevant here.
                    # So a single pass of adding all possible non-conflicting elements is safe.
            
            if improved:
                no_improve_count = 0
                if current_size > best_size:
                    best_size = current_size
                    best_set = list(current_set)
                # After adding, we might have new opportunities for swaps? 
                # Swaps are about replacing one element to allow another.
                # If we added freely, we are in a "maximal" state w.r.t single additions.
                # Now we need swaps to escape local maxima.
            
            # Phase 2: Try swaps to enable additions
            # We look for elements `u` not in set that conflict with exactly one element `v` in set.
            # If we swap `v` with `u`, we might be able to add other elements.
            
            # To speed this up, we can precompute conflict counts for elements not in set?
            # That's expensive. 
            # Instead, we iterate through not_in_set again (or a subset).
            
            # Optimization: Only try swaps if we are stuck (no additions in Phase 1)
            if not improved:
                no_improve_count += 1
                
                # If stuck for a while, try random restarts within local search
                # or perturb the current set.
                if no_improve_count > max_no_improve:
                    # Perturbation: Remove 1-2 random elements from current_set
                    # This creates space.
                    if current_size > 2:
                        remove_count = min(2, current_size)
                        to_remove = random.sample(list(current_set), remove_count)
                        for r in to_remove:
                            current_set.remove(r)
                        current_size -= remove_count
                        no_improve_count = 0
                        # After perturbation, go back to Phase 1
                        continue
                    else:
                        # If set is very small, break and restart from scratch
                        break

                # Try swaps
                # For efficiency, we sample a subset of not_in_set
                # Or iterate all. m=205 is small.
                not_in_set = [i for i in range(m) if i not in current_set]
                random.shuffle(not_in_set)
                
                swap_made = False
                for u in not_in_set:
                    if time.time() > time_limit:
                        break
                    
                    # Find conflicts of u in current_set
                    conflicts = []
                    for v in current_set:
                        if (u - v) % m in forbidden:
                            conflicts.append(v)
                    
                    # We only consider swaps if u conflicts with exactly 1 element
                    # If it conflicts with 0, Phase 1 would have added it.
                    # If it conflicts with >1, swapping 1 element doesn't fix it.
                    if len(conflicts) == 1:
                        v = conflicts[0]
                        
                        # Perform swap: remove v, add u
                        current_set.remove(v)
                        current_set.add(u)
                        
                        # Now check if this swap allows adding other elements
                        # We need to check if any element w (not in set, w != u) 
                        # is now addable.
                        # Actually, we can just run Phase 1 logic on the new set.
                        # But we need to be able to revert if no improvement.
                        
                        # Let's try to add elements
                        added_any = False
                        # We can't just iterate all m again easily without overhead.
                        # Let's check if u's addition "unblocks" anything?
                        # No, u is in the set. 
                        # The removal of v might unblock elements that conflicted with v.
                        
                        # Let's try to add any element that was previously blocked by v?
                        # That's complex to track. 
                        # Simple approach: 
                        # Check if we can add any element from not_in_set (excluding u) 
                        # that has 0 conflicts with the *new* current_set.
                        
                        # To be fast, we only check a few candidates?
                        # Or just check all?
                        # m is 205. 205 * 205 = 42k ops. Fast enough.
                        
                        new_not_in_set = [i for i in range(m) if i not in current_set]
                        random.shuffle(new_not_in_set)
                        
                        for w in new_not_in_set:
                            if can_add(w, current_set):
                                current_set.add(w)
                                current_size += 1
                                added_any = True
                                # We can try to add more, but let's break for now
                                # to keep the swap atomic-ish?
                                # Or continue adding?
                                # If we add w, we might block others.
                                # Let's just add one and see if size increased.
                                break
                        
                        if added_any:
                            # Improvement!
                            improved = True
                            swap_made = True
                            if current_size > best_size:
                                best_size = current_size
                                best_set = list(current_set)
                            no_improve_count = 0
                            break # Restart Phase 1
                        else:
                            # No improvement, revert swap
                            current_set.remove(u)
                            current_set.add(v)
                            # Size remains same
                            # Try next u
                
                if swap_made:
                    continue # Go back to Phase 1
            
            # If we got here, no improvement in this iteration
            # Check time
            if time.time() > time_limit:
                break

        return best_set

    best_set = []
    best_size = 0
    
    # Run multiple restarts
    # Estimate time per restart: 
    # Greedy: O(m^2) ~ 40k ops.
    # Local search: Variable.
    # Let's run until deadline - 5s.
    
    num_restarts = 0
    while time.time() < deadline - 5:
        num_restarts += 1
        init_set = greedy_init()
        
        # Allocate more time to local search if we are finding good solutions?
        # Or just give each restart a fixed chunk of time?
        # If we give fixed time, we might finish early.
        # Let's give each local search a budget of 1 second?
        # 580 seconds / 1 sec = 580 restarts.
        # This seems reasonable.
        
        ls_start = time.time()
        ls_budget = min(1.0, deadline - 5 - ls_start)
        if ls_budget <= 0:
            break
            
        ls_set = local_search(init_set, ls_start + ls_budget)
        
        if len(ls_set) > best_size:
            best_size = len(ls_set)
            best_set = ls_set
            
        # If we found a very good solution, maybe do a longer search on it?
        # For now, uniform restarts.

    # Final verification
    if not best_set:
        return np.array([0], dtype=np.int64)
        
    best_set = list(set(best_set))
    
    # Verify validity
    valid = True
    for i in range(len(best_set)):
        for j in range(i+1, len(best_set)):
            d = (best_set[i] - best_set[j]) % m
            if d in forbidden:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to a known valid small set
        # Greedy on 0..204
        fallback = []
        for i in range(m):
            ok = True
            for s in fallback:
                if (i - s) % m in forbidden:
                    ok = False
                    break
            if ok:
                fallback.append(i)
        return np.array(sorted(fallback), dtype=np.int64)
        
    return np.array(sorted(best_set), dtype=np.int64)

# EVOLVE-BLOCK-END
