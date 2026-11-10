import time
import random
import sys

N = 256

# EVOLVE-BLOCK-START

def _can_add(x, selected, sums):
    """Check if x can be added to the current Sidon set."""
    if 2 * x in sums:
        return False
    for a in selected:
        if x + a in sums:
            return False
    return True

def _greedy_maximize(selected, order, sums=None):
    """
    Greedily adds elements from 'order' to 'selected' to make it maximal.
    Returns the new selected list and sums set.
    """
    if sums is None:
        sums = set()
        for i in range(len(selected)):
            for j in range(i, len(selected)):
                sums.add(selected[i] + selected[j])
    
    new_selected = list(selected)
    new_sums = set(sums)
    
    for x in order:
        if x in new_sums: # Optimization: if x is already a sum, it's blocked? No, x being a sum doesn't block x directly, 
                          # but if x+a is a sum, it blocks. 
                          # Actually, if x is in the set, we skip.
            pass
        
        # Check if x is already in selected
        # Using a set for membership is faster, but we need to keep the list order? 
        # For Sidon property, order doesn't matter. For maximality, we just need the set.
        # However, _can_add iterates over selected.
        
        # Quick check: is x in new_selected?
        # Since we are adding from an order of [1..N], and selected is a subset, 
        # we can maintain a boolean array or set for fast lookup.
        
        # Let's use a set for fast lookup of elements in the set.
        # But _can_add needs to check sums.
        
        # To be safe and fast, let's just check if x is in the set of selected elements.
        # We'll maintain a set `sel_set` alongside.
        
        # Wait, the function signature takes `selected` list. 
        # Let's optimize _can_add to take a set as well? 
        # Or just assume `order` contains elements not in `selected`?
        # The caller should filter.
        
        # For now, let's assume the caller passes a list of candidates not in selected.
        # But to be robust:
        
        # Actually, a simpler approach for the SA step:
        # We have a maximal set S.
        # We remove an element y.
        # Now S' = S \ {y} is NOT maximal.
        # We want to re-maximize S' using a greedy strategy.
        # The candidates are [1..N] \ S'.
        # We want to add as few as possible? No, greedy adds if possible.
        # To minimize size, we want a greedy order that adds "blocking" elements first.
        # But standard greedy adds if possible.
        
        # Let's implement a helper that takes a set and a list of candidates and returns the new maximal set.
        
        # Optimization: Use a boolean array for "in_set" to avoid set lookups if needed, 
        # but set lookups in Python are O(1).
        
        # Let's refactor to use sets for elements too.
        
        # Re-implementation of the inner loop for speed:
        # We need to check if x can be added.
        # Conflict if (2x in sums) or (any x+a in sums for a in selected).
        
        # We can precompute which x are blocked by current sums?
        # If s in sums, then x is blocked if x = s/2 (and s even) or if there exists a in selected such that x+a=s => x = s-a.
        
        # This is getting complex. Let's stick to the simple O(|S|) check per candidate, 
        # but optimize the loop.
        
        # Since this is called inside SA, speed is critical.
        
        # Let's use a list of booleans for "blocked_by_sums" ?
        # No, blocked status changes as we add elements.
        
        # Let's just do the check.
        
        # Is x in new_selected?
        # We'll maintain a set `new_sel_set` for O(1) check.
        
        # Wait, I didn't define new_sel_set. Let's do it properly.
        
        pass # Placeholder for actual logic below

def _greedy_maximize_fast(initial_set, order):
    """
    Takes a set (or list) of initial elements and a list of candidate elements.
    Greedily adds elements from 'order' if they can be added to the Sidon set.
    Returns (final_list, final_sums_set).
    """
    selected = list(initial_set)
    sel_set = set(selected)
    
    sums = set()
    # Compute initial sums
    for i in range(len(selected)):
        a = selected[i]
        sums.add(2*a)
        for j in range(i+1, len(selected)):
            sums.add(a + selected[j])
            
    # We want to minimize the final size. 
    # Greedy addition: if we can add x, we add it.
    # The order of 'order' determines which elements get added.
    # To get a *small* maximal set, we want to add elements that block *many* other elements.
    # However, standard greedy just adds if possible.
    # Different orders yield different maximal sets.
    
    for x in order:
        if x in sel_set:
            continue
            
        # Check if x can be added
        if 2*x in sums:
            continue
            
        # Check conflicts with existing elements
        conflict = False
        for a in selected:
            if x + a in sums:
                conflict = True
                break
                
        if conflict:
            continue
            
        # Add x
        selected.append(x)
        sel_set.add(x)
        sums.add(2*x)
        for a in selected[:-1]: # Only need to check against previous elements? 
                                # x + a for all a in selected (including x? No, 2x already handled)
                                # We need to add x+a for all a in selected BEFORE adding x?
                                # No, x+a for a in old_selected.
                                # And x+x = 2x.
                                # We already added 2x.
                                # So add x+a for a in old_selected.
                                # But 'selected' now contains x.
                                # So iterate over selected[:-1].
                                pass
        
        # Correction: 
        # When adding x, new sums are {2x} U {x+a | a in old_selected}.
        # We already added 2x.
        # Now add x+a for a in old_selected.
        # Note: 'selected' was appended x, so old_selected is selected[:-1].
        for a in selected[:-1]:
            sums.add(x + a)
            
    return selected, sums

def _is_valid_solution(arr):
    """
    Verifies Sidon and Maximality.
    """
    if not arr:
        return False
    
    arr_set = set(arr)
    if len(arr_set) != len(arr):
        return False
    
    for x in arr:
        if x < 1 or x > N:
            return False
            
    # Compute sums
    sums = set()
    for i in range(len(arr)):
        a = arr[i]
        if 2*a in sums:
            return False
        sums.add(2*a)
        for j in range(i+1, len(arr)):
            b = arr[j]
            s = a + b
            if s in sums:
                return False
            sums.add(s)
            
    # Check maximality
    for x in range(1, N + 1):
        if x in arr_set:
            continue
        
        # If x can be added, it's not maximal.
        # x can be added if 2x not in sums AND no x+a in sums for a in arr.
        
        if 2*x in sums:
            continue
            
        can_add = True
        for a in arr:
            if x + a in sums:
                can_add = False
                break
                
        if can_add:
            return False
            
    return True

def construct():
    start_time = time.time()
    # Safety margin: 55s limit (5s buffer for overhead)
    deadline = start_time + 55.0
    
    rng = random.Random(42)
    
    # Initial solution: Greedy from random order
    # We want a small set. 
    # Random greedy often produces sets of size ~20-25 for N=256.
    # We want to minimize this.
    
    order = list(range(1, N + 1))
    rng.shuffle(order)
    current_set, current_sums = _greedy_maximize_fast([], order)
    
    best_set = current_set[:]
    best_size = len(current_set)
    
    # Simulated Annealing
    # State: a maximal Sidon set.
    # Move: Remove a random element y from the set.
    #       Then re-maximize the resulting set using a random greedy order.
    #       The new set is maximal.
    #       Size change: New size vs Old size.
    #       Cost = New Size - Old Size.
    #       We want to minimize Size. So we accept if Cost < 0 (size decreased).
    #       If Cost > 0 (size increased), we accept with probability exp(-Cost/T).
    
    # Note: Removing one element and re-maximizing might add many elements back,
    # or just one, or none.
    # If we remove y, and y was "critical" for blocking many elements,
    # re-maximizing might add many new elements to block them, increasing size.
    # If y was "redundant" (its sums were covered by others), removing it and re-maximizing
    # might result in a set of the same size (if a different element blocks the same things)
    # or smaller size (if the re-maximization finds a more efficient blocking element).
    
    # Actually, if we remove y, the set is no longer maximal.
    # The greedy re-maximization will add elements until maximal.
    # It is possible that the new maximal set is smaller than the old one.
    
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.995
    
    # Precompute a random order for re-maximization? 
    # Or use a fresh random order each time?
    # Fresh random order provides diversity.
    
    # Optimization: 
    # Instead of full re-maximization from scratch (which is O(N * |S|)),
    # we can just add the elements that were blocked by y?
    # No, removing y might unblock elements that were previously blocked by y,
    # but also, the sums change? No, sums of remaining elements are unchanged.
    # Only the "blockage" status of candidates changes.
    # Candidates that were blocked solely by y might now be addable.
    # Candidates that were blocked by other elements are still blocked.
    # So, we only need to consider candidates that were blocked by y.
    # How to find them?
    # x was blocked if 2x in sums_old or (x+a in sums_old for some a in S_old).
    # Since y is removed, sums_new = sums_old \ {sums involving y}.
    # This is complex to track incrementally.
    
    # Given the 55s limit and N=256, full re-maximization is probably fast enough.
    # |S| is ~20. N is 256.
    # Cost of re-max: 256 * 20 = 5120 operations.
    # We can do ~1000 moves in 1 second? Maybe less in Python.
    # Let's try.
    
    current_size = len(current_set)
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Pick a random element to remove
        idx = rng.randint(0, len(current_set) - 1)
        y = current_set[idx]
        
        # Create new initial set (current_set without y)
        new_initial = [x for i, x in enumerate(current_set) if i != idx]
        
        # Generate random order for re-maximization
        # To encourage finding smaller sets, maybe sort candidates by "potential to block"?
        # For now, random shuffle.
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        new_set, _ = _greedy_maximize_fast(new_initial, order)
        new_size = len(new_set)
        
        delta = new_size - current_size
        
        if delta < 0:
            # Accept
            current_set = new_set
            current_size = new_size
            if current_size < best_size:
                best_size = current_size
                best_set = current_set[:]
                # Optional: restart SA with higher T? Or just continue?
                # Restarting might be good to explore new basins.
                # But let's just continue cooling for now.
        else:
            # Accept with probability exp(-delta / T)
            # delta is positive
            prob = 2.718281828 ** (-delta / T)
            if rng.random() < prob:
                current_set = new_set
                current_size = new_size
        
        T *= cooling_rate
        
    # Final check
    if not _is_valid_solution(best_set):
        # Fallback
        order = list(range(1, N + 1))
        rng.shuffle(order)
        best_set, _ = _greedy_maximize_fast([], order)
        
    return best_set

# EVOLVE-BLOCK-END
