# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 150s kill (15s buffer)
    
    MAX_VAL = 1000
    MAX_SUM = 2 * MAX_VAL

    def _validate(arr):
        """
        Validate that arr is a valid B2[2] set.
        Returns True if valid, False otherwise.
        """
        counts = [0] * (MAX_SUM + 1)
        n = len(arr)
        for i in range(n):
            ai = arr[i]
            for j in range(i, n):
                s = ai + arr[j]
                counts[s] += 1
                if counts[s] > 2:
                    return False
        return True

    def _greedy(order):
        """
        Construct a B2[2] set by processing candidates in the given order.
        Returns the set as a list.
        """
        chosen = []
        counts = [0] * (MAX_SUM + 1)
        
        for x in order:
            # Check if adding x violates the B2[2] condition
            # New pairs: {x,x} with sum 2x, and {x,y} for all y in chosen
            valid = True
            
            # Check 2*x
            if counts[2 * x] >= 2:
                valid = False
            else:
                # Check x + y for all y in chosen
                for y in chosen:
                    s = x + y
                    if counts[s] >= 2:
                        valid = False
                        break
            
            if valid:
                chosen.append(x)
                # Update counts for new pairs involving x
                counts[2 * x] += 1
                for y in chosen[:-1]:  # All previously chosen elements
                    s = x + y
                    counts[s] += 1
        
        return chosen

    def _local_improve(initial_set, time_limit):
        """
        Try to improve the set by removing elements and trying to add others.
        Uses incremental count updates for speed.
        """
        chosen = list(initial_set)
        chosen_set = set(chosen)
        
        # Build initial counts
        counts = [0] * (MAX_SUM + 1)
        for i in range(len(chosen)):
            for j in range(i, len(chosen)):
                s = chosen[i] + chosen[j]
                counts[s] += 1
        
        available = set(range(1, MAX_VAL + 1)) - chosen_set
        
        improved = True
        best_size = len(chosen)
        
        while improved and time.time() < time_limit:
            improved = False
            # Shuffle chosen to try different removal orders
            indices = list(range(len(chosen)))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() >= time_limit:
                    break
                    
                x = chosen[idx]
                
                # 1. Remove x
                # Decrement counts for pairs involving x
                counts[2 * x] -= 1
                for y in chosen:
                    if y == x:
                        continue
                    s = x + y
                    counts[s] -= 1
                
                chosen_set.remove(x)
                # Remove x from list (keeping it unsorted for now, or we can keep it sorted)
                # For efficiency in the "add" loop, having a list is good. 
                # Let's pop it.
                chosen.pop(idx)
                
                # 2. Try to add available elements
                # We want to add as many as possible, or at least net gain > 0
                # Since we removed 1, we need to add >= 2 to improve size.
                # But even adding 1 might allow further additions later? 
                # Actually, removing 1 and adding 1 is a swap. 
                # Removing 1 and adding 2 is a net gain.
                
                # Let's try to greedily add available elements in random or sorted order
                # To maximize chances of adding multiple, we might want to try smaller numbers first 
                # or random. Let's try random to explore.
                
                candidates = list(available)
                random.shuffle(candidates)
                
                added = 0
                new_chosen = chosen[:]
                new_chosen_set = chosen_set.copy()
                new_counts = counts[:] # Copy counts for this trial
                
                # We need to be careful: if we add y, it affects counts for subsequent z.
                # So we simulate the additions on new_counts.
                
                for y in candidates:
                    if y in new_chosen_set:
                        continue
                    
                    # Check if y can be added to new_chosen
                    valid = True
                    if new_counts[2 * y] >= 2:
                        valid = False
                    else:
                        for z in new_chosen:
                            s = y + z
                            if new_counts[s] >= 2:
                                valid = False
                                break
                    
                    if valid:
                        # Add y to the trial set
                        new_chosen.append(y)
                        new_chosen_set.add(y)
                        new_counts[2 * y] += 1
                        for z in new_chosen[:-1]: # z is in new_chosen before y was appended
                            s = y + z
                            new_counts[s] += 1
                        added += 1
                        # If we have already gained size (added > 1), we could stop early?
                        # No, adding more might be possible. But if we added 1, we are at break-even size.
                        # If we added 2, we are at +1 size.
                        # Let's just run through all candidates to maximize the local gain.
                        # However, this is O(N * M) where M is available.
                        # To speed up, we can break if added > 1 and we've tried enough?
                        # Let's just let it run, but check time.
                        if time.time() > time_limit:
                            break
                
                if added > 1: # Net gain
                    # Commit the changes
                    chosen = new_chosen
                    chosen_set = new_chosen_set
                    counts = new_counts
                    available = set(range(1, MAX_VAL + 1)) - chosen_set
                    best_size = len(chosen)
                    improved = True
                    # Break the inner loop to restart the outer loop with the new state
                    break
                else:
                    # Roll back: We already popped x from chosen. We need to put it back?
                    # Actually, the loop structure above modified `chosen` in place by popping.
                    # If we don't commit, we must restore `chosen` and `counts`.
                    # This is getting complicated with in-place modifications.
                    
                    # Let's restructure: Do the removal/addition on copies, and only commit if successful.
                    # The current code block above did `chosen.pop(idx)` BEFORE the trial.
                    # So if `added <= 1`, we must restore.
                    
                    # Restore x
                    chosen.insert(idx, x)
                    chosen_set.add(x)
                    
                    # Restore counts
                    # We modified `counts` in place at the start of the loop (decrements).
                    # We did NOT modify `counts` during the trial (we used `new_counts`).
                    # So we just need to increment `counts` back for x.
                    counts[2 * x] += 1
                    for y in chosen:
                        if y == x:
                            continue
                        s = x + y
                        counts[s] += 1
                    # Wait, the restoration of counts is tricky because `chosen` list order might have changed?
                    # No, we inserted x back at idx. The set of elements is the same.
                    # The counts logic:
                    # Original counts C.
                    # Step 1: C' = C - pairs(x).
                    # Step 2: Trial on C'.
                    # Step 3: If fail, restore C.
                    # My code above:
                    # 1. `counts` (global to this function) was decremented.
                    # 2. `new_counts` was a copy.
                    # 3. If fail, we need to add back the pairs of x to `counts`.
                    #    The loop `for y in chosen: if y != x: counts[x+y] += 1` does this.
                    #    And `counts[2*x] += 1`.
                    #    This correctly restores `counts`.
                    
                    # Note: `available` set was not modified during the trial (we used `candidates` from it, but didn't remove from it).
                    # So `available` is still valid.
        
        return chosen

    # 1. Initial deterministic greedy
    initial_order = list(range(1, MAX_VAL + 1))
    best_set = _greedy(initial_order)
    best_size = len(best_set)
    
    # 2. Local improvement on initial set
    if time.time() < deadline - 10:
        improved_set = _local_improve(best_set, deadline - 10)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_set = improved_set

    # 3. Randomized restarts
    random.seed(42)
    while time.time() < deadline - 20:
        # Random permutation
        order = list(range(1, MAX_VAL + 1))
        random.shuffle(order)
        
        current_set = _greedy(order)
        current_size = len(current_set)
        
        if current_size > best_size:
            best_size = current_size
            best_set = current_set
        
        # Try local improvement if we have time and the set is promising
        if time.time() < deadline - 15:
            # Limit the time for local improvement per restart
            local_time_limit = min(deadline - 15, time.time() + 5.0)
            improved = _local_improve(current_set, local_time_limit)
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved

    return best_set

# EVOLVE-BLOCK-END
