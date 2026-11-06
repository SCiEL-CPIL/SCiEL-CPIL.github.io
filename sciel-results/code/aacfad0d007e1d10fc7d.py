import random
from fractions import Fraction
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Precompute reciprocal values
    vals = [Fraction(1, d) for d in denominators]
    
    # Best solution found so far
    best_indices = []
    best_sum = Fraction(0)
    best_deficit = Fraction(1, 1) # Initial deficit is 1 (empty set)
    
    def update_best(indices, s):
        nonlocal best_indices, best_sum, best_deficit
        if s >= 1:
            return
        deficit = 1 - s
        if deficit < best_deficit:
            best_deficit = deficit
            best_sum = s
            best_indices = list(indices)
            
    def greedy_init(order):
        """
        Generate an initial solution by adding items in the given order
        if it keeps the sum < 1.
        """
        current_indices = []
        current_sum = Fraction(0)
        for i in order:
            if current_sum + vals[i] < 1:
                current_indices.append(i)
                current_sum += vals[i]
        return current_indices, current_sum

    def local_search(initial_indices, initial_sum):
        """
        Perform local search starting from the given state.
        Returns the best (indices, sum) found.
        """
        current_indices = list(initial_indices)
        current_sum = initial_sum
        current_set = set(current_indices)
        
        # Update global best with initial state
        update_best(current_indices, current_sum)
        
        improved = True
        while improved:
            improved = False
            
            # Check time
            if time.time() - start_time > DEADLINE:
                break
                
            # 1. Try adding a single unused element
            # We iterate through all unused elements to find the best add
            best_add_idx = -1
            best_add_sum = current_sum
            best_add_deficit = 1 - current_sum
            
            for i in range(n):
                if i in current_set:
                    continue
                new_sum = current_sum + vals[i]
                if new_sum < 1:
                    new_deficit = 1 - new_sum
                    if new_deficit < best_add_deficit:
                        best_add_deficit = new_deficit
                        best_add_sum = new_sum
                        best_add_idx = i
                        
            if best_add_idx != -1:
                current_indices.append(best_add_idx)
                current_sum = best_add_sum
                current_set.add(best_add_idx)
                improved = True
                update_best(current_indices, current_sum)
                continue # Restart loop to re-evaluate other moves from new state

            # 2. Try removing a single used element
            # Removing an element increases the deficit, so this is only useful
            # if it allows a subsequent better addition? 
            # Actually, pure removal is never beneficial for minimizing deficit 
            # if we can just keep the item. However, in local search, 
            # we might need to remove to make space for a swap or 2-for-1.
            # But direct removal increases deficit, so we skip pure removal 
            # unless it's part of a swap logic which we handle below.
            # Wait, if we remove an item, the deficit increases. 
            # So we only remove if we add something else.
            # So "Remove" alone is not a move that improves the objective.
            # We skip pure removal.

            # 3. Try swapping a used element with an unused one
            best_swap_remove = -1
            best_swap_add = -1
            best_swap_sum = current_sum
            best_swap_deficit = 1 - current_sum
            
            for i in current_indices:
                for j in range(n):
                    if j in current_set:
                        continue
                    # Remove i, Add j
                    new_sum = current_sum - vals[i] + vals[j]
                    if new_sum < 1:
                        new_deficit = 1 - new_sum
                        if new_deficit < best_swap_deficit:
                            best_swap_deficit = new_deficit
                            best_swap_sum = new_sum
                            best_swap_remove = i
                            best_swap_add = j
            
            if best_swap_remove != -1:
                # Perform swap
                current_indices.remove(best_swap_remove)
                current_indices.append(best_swap_add)
                current_sum = best_swap_sum
                current_set.remove(best_swap_remove)
                current_set.add(best_swap_add)
                improved = True
                update_best(current_indices, current_sum)
                continue # Restart loop

            # 4. Try 2-for-1 swap: Remove 1, Add 2
            # This is more expensive O(N^3) but can escape local optima.
            best_21_remove = -1
            best_21_add1 = -1
            best_21_add2 = -1
            best_21_sum = current_sum
            best_21_deficit = 1 - current_sum
            
            # To optimize, we can limit the number of 2-for-1 checks 
            # or break early if we find a good one.
            # N=47, N^3 is ~100k. This is fast enough for a few passes.
            
            for i in current_indices:
                # Remove i
                base_sum = current_sum - vals[i]
                # Try adding j, k
                # We can iterate j, then k
                # To avoid duplicates (j,k) and (k,j), we can enforce j < k?
                # No, indices are distinct, but values are distinct.
                # We just need distinct indices.
                
                # Optimization: Only consider j, k that are not in current_set
                # and j != i, k != i.
                
                # Iterate over unused elements
                unused = [x for x in range(n) if x not in current_set]
                
                # If fewer than 2 unused, skip
                if len(unused) < 2:
                    continue
                    
                # Iterate pairs from unused
                # To save time, we can just pick the best pair
                # But checking all pairs is O(U^2) where U is unused count.
                # Total O(N * U^2).
                
                for u_idx1 in range(len(unused)):
                    j = unused[u_idx1]
                    val_j = vals[j]
                    if base_sum + val_j >= 1:
                        continue
                    for u_idx2 in range(u_idx1 + 1, len(unused)):
                        k = unused[u_idx2]
                        val_k = vals[k]
                        new_sum = base_sum + val_j + val_k
                        if new_sum < 1:
                            new_deficit = 1 - new_sum
                            if new_deficit < best_21_deficit:
                                best_21_deficit = new_deficit
                                best_21_sum = new_sum
                                best_21_remove = i
                                best_21_add1 = j
                                best_21_add2 = k
            
            if best_21_remove != -1:
                # Perform 2-for-1 swap
                current_indices.remove(best_21_remove)
                current_indices.append(best_21_add1)
                current_indices.append(best_21_add2)
                current_sum = best_21_sum
                current_set.remove(best_21_remove)
                current_set.add(best_21_add1)
                current_set.add(best_21_add2)
                improved = True
                update_best(current_indices, current_sum)
                continue # Restart loop

            # 5. Try 2-for-2 swap: Remove 2, Add 2
            # This is O(N^4) which is ~5M. Might be too slow for many passes.
            # We will only do this if we are stuck and have time.
            if time.time() - start_time < DEADLINE - 10:
                best_22_remove1 = -1
                best_22_remove2 = -1
                best_22_add1 = -1
                best_22_add2 = -1
                best_22_sum = current_sum
                best_22_deficit = 1 - current_sum
                
                used = list(current_indices)
                unused = [x for x in range(n) if x not in current_set]
                
                if len(used) >= 2 and len(unused) >= 2:
                    # Iterate pairs from used
                    for u_idx1 in range(len(used)):
                        i1 = used[u_idx1]
                        for u_idx2 in range(u_idx1 + 1, len(used)):
                            i2 = used[u_idx2]
                            base_sum = current_sum - vals[i1] - vals[i2]
                            
                            # Iterate pairs from unused
                            for v_idx1 in range(len(unused)):
                                j1 = unused[v_idx1]
                                val_j1 = vals[j1]
                                if base_sum + val_j1 >= 1:
                                    continue
                                for v_idx2 in range(v_idx1 + 1, len(unused)):
                                    j2 = unused[v_idx2]
                                    val_j2 = vals[j2]
                                    new_sum = base_sum + val_j1 + val_j2
                                    if new_sum < 1:
                                        new_deficit = 1 - new_sum
                                        if new_deficit < best_22_deficit:
                                            best_22_deficit = new_deficit
                                            best_22_sum = new_sum
                                            best_22_remove1 = i1
                                            best_22_remove2 = i2
                                            best_22_add1 = j1
                                            best_22_add2 = j2
                                                    
                if best_22_remove1 != -1:
                    current_indices.remove(best_22_remove1)
                    current_indices.remove(best_22_remove2)
                    current_indices.append(best_22_add1)
                    current_indices.append(best_22_add2)
                    current_sum = best_22_sum
                    current_set.remove(best_22_remove1)
                    current_set.remove(best_22_remove2)
                    current_set.add(best_22_add1)
                    current_set.add(best_22_add2)
                    improved = True
                    update_best(current_indices, current_sum)
                    continue # Restart loop

        return current_indices, current_sum

    # Restart strategy
    # We will try several different initial orderings.
    
    # 1. Sorted by value descending (Greedy largest first)
    order1 = sorted(range(n), key=lambda i: vals[i], reverse=True)
    idx, s = greedy_init(order1)
    local_search(idx, s)
    
    # 2. Sorted by value ascending (Greedy smallest first - usually worse sum, but different basin)
    if time.time() - start_time < DEADLINE:
        order2 = sorted(range(n), key=lambda i: vals[i])
        idx, s = greedy_init(order2)
        local_search(idx, s)
        
    # 3. Random restarts
    restarts = 0
    while time.time() - start_time < DEADLINE - 5:
        order = list(range(n))
        random.shuffle(order)
        idx, s = greedy_init(order)
        local_search(idx, s)
        restarts += 1
        # If we've done many restarts, we might be stuck in a good local optimum.
        # We can break if we haven't improved in a while? 
        # The local_search updates global best. 
        # We just keep going until time runs out.
        
    return best_indices

