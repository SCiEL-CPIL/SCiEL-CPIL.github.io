import time
from collections import Counter

def construct():
    start_time = time.time()
    DEADLINE = 115.0 # Safety margin
    
    # 1. Greedy Construction
    chosen = []
    counts = [0] * 2001 # Index 0..2000
    
    for x in range(1, 1001):
        # Check if x can be added
        valid = True
        # Sums involving x: x+x, x+y for y in chosen
        # We need to check if adding these makes any count > 2
        # Note: x+x is one sum. x+y is distinct if y != x.
        # Since y < x (as we iterate increasing), x+y is unique for each y.
        
        # Optimization: Pre-check max possible
        # We just simulate
        new_counts = counts[:] # Copying list of 2001 ints is fast enough? 
        # 2001 * 1000 iterations = 2M ops. Acceptable.
        
        # Actually, we can just check the specific sums
        # Sums are x+x and x+y for y in chosen
        # If any of these sums currently has count 2, adding x makes it 3 -> Invalid.
        # If any has count 1, it becomes 2 -> OK.
        # If any has count 0, it becomes 1 -> OK.
        
        if counts[2*x] == 2:
            continue
            
        can_add = True
        for y in chosen:
            s = x + y
            if counts[s] == 2:
                can_add = False
                break
        
        if can_add:
            # Commit
            chosen.append(x)
            counts[2*x] += 1
            for y in chosen[:-1]: # y < x
                counts[x+y] += 1
    
    best_size = len(chosen)
    best_set = set(chosen)
    
    # 2. Local Search
    # We represent the set as a list for iteration, and a set for membership.
    # We maintain counts.
    
    # To allow swaps, we need to be able to remove elements.
    # Current state: chosen list, counts array.
    
    # Helper to check if adding v is valid given current counts and chosen
    def try_add(v, chosen_list, counts_arr):
        if counts_arr[2*v] == 2:
            return False
        for y in chosen_list:
            if y == v: continue # v not in set
            s = v + y
            if counts_arr[s] == 2:
                return False
        return True

    def do_add(v, chosen_list, counts_arr):
        counts_arr[2*v] += 1
        for y in chosen_list:
            if y == v: continue
            counts_arr[v+y] += 1
        chosen_list.append(v)

    def try_swap(u, v, chosen_list, counts_arr):
        # Remove u, add v
        # 1. Check if removing u and adding v keeps counts <= 2
        # Sums affected:
        # Removed: u+u, u+y for y in chosen, y!=u
        # Added: v+v, v+y for y in chosen_new (which is chosen - {u} + {v})
        # Note: v+y where y=v is v+v.
        # v+y where y != v, y in chosen_new.
        
        # Let's simulate
        # Temporarily remove u
        # We can't modify counts_arr directly if we want to revert easily, 
        # but we can use a delta approach or just copy if small.
        # Copying 2001 ints is fast.
        
        temp_counts = counts_arr[:]
        
        # Remove u
        temp_counts[2*u] -= 1
        for y in chosen_list:
            if y == u: continue
            temp_counts[u+y] -= 1
            
        # Add v
        if temp_counts[2*v] == 2: # Becomes 3
            return False
        temp_counts[2*v] += 1
        
        for y in chosen_list:
            if y == u: continue # u is removed
            if y == v: continue # v is not in chosen yet, but v+v handled
            # Check sum v+y
            if temp_counts[v+y] == 2:
                return False
            temp_counts[v+y] += 1
            
        return True

    def do_swap(u, v, chosen_list, counts_arr):
        # Remove u
        counts_arr[2*u] -= 1
        for y in chosen_list:
            if y == u: continue
            counts_arr[u+y] -= 1
        chosen_list.remove(u)
        
        # Add v
        counts_arr[2*v] += 1
        for y in chosen_list:
            if y == v: continue
            counts_arr[v+y] += 1
        chosen_list.append(v)

    # Local Search Loop
    # We want to increase size.
    # Strategy:
    # 1. Try to add any element not in set.
    # 2. If no add, try swaps to find a configuration that allows an add.
    
    # To avoid O(N^3), we will limit the scope or use a smarter search.
    # Given the 120s limit, we can afford several rounds of O(N^2) checks.
    
    current_list = chosen[:]
    current_counts = counts[:]
    current_set = set(current_list)
    
    improved = True
    while improved and (time.time() - start_time) < DEADLINE:
        improved = False
        
        # Phase 1: Try to add
        # Iterate through candidates not in set
        # We can iterate 1..1000 and skip those in current_set
        for v in range(1, 1001):
            if v in current_set:
                continue
            if try_add(v, current_list, current_counts):
                do_add(v, current_list, current_counts)
                current_set.add(v)
                if len(current_list) > best_size:
                    best_size = len(current_list)
                    best_set = set(current_list)
                improved = True
                break # Restart to try more adds
        
        if improved:
            continue # Restart outer loop
        
        # Phase 2: Try swaps
        # If we can't add, we try to swap.
        # We want a swap that enables an add.
        # We will iterate through u in current_list and v not in current_set.
        # To save time, we might not check all pairs if it's too slow.
        # But 450*550 = 247k checks. Each check is O(450). 110M ops.
        # In Python, this might take 5-10 seconds.
        # We should only do this if we have time.
        
        if (time.time() - start_time) > DEADLINE - 10:
            break
            
        # Shuffle to avoid bias? Or just ordered?
        # Ordered might be deterministic. Let's try ordered first.
        found_swap = False
        # To make it faster, we can break after first valid swap that leads to an add?
        # No, we just do the swap, then restart Phase 1.
        
        for u in current_list:
            if (time.time() - start_time) > DEADLINE - 5:
                break
            for v in range(1, 1001):
                if v in current_set:
                    continue
                if u == v:
                    continue
                if try_swap(u, v, current_list, current_counts):
                    # Perform swap
                    do_swap(u, v, current_list, current_counts)
                    current_set.remove(u)
                    current_set.add(v)
                    improved = True
                    found_swap = True
                    break
            if found_swap:
                break
                
    return list(best_set)
