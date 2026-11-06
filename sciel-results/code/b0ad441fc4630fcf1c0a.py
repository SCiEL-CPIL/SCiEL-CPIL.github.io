import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    fixed_elements = [1, 2, 5, 11]
    fixed_set = set(fixed_elements)
    N = 284
    
    def get_initial_state(target_size):
        """
        Creates a random set of size target_size containing fixed elements.
        Returns A (list), counts (dict), collisions (int).
        """
        # Start with fixed elements
        A = list(fixed_elements)
        # Add random elements
        candidates = [x for x in range(1, N + 1) if x not in fixed_set]
        random.shuffle(candidates)
        
        # We need target_size - len(fixed_elements) more elements
        num_to_add = target_size - len(fixed_elements)
        if num_to_add > 0:
            A.extend(candidates[:num_to_add])
        
        # If we don't have enough candidates (unlikely here), fill with 1s? No, distinct required.
        # But N=284 is large enough.
        
        # Calculate sums and collisions
        counts = {}
        for i in range(len(A)):
            for j in range(i, len(A)):
                s = A[i] + A[j]
                counts[s] = counts.get(s, 0) + 1
        
        collisions = sum(1 for c in counts.values() if c > 1)
        return A, counts, collisions

    def simulate_annealing(target_size, max_steps=50000, start_temp=10.0, end_temp=0.01):
        """
        Attempts to find a valid Sidon set of size target_size.
        Returns A if found, else None.
        """
        A, counts, collisions = get_initial_state(target_size)
        
        # Identify variable indices (not fixed)
        # We will swap elements at these indices
        # To make it simple, we just treat the list A as the state.
        # Fixed elements are at indices 0, 1, 2, 3.
        # We only swap indices >= 4.
        
        current_temp = start_temp
        cooling_rate = (end_temp / start_temp) ** (1.0 / max_steps)
        
        best_A = None
        best_collisions = collisions
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Pick a variable index to swap
            if len(A) > 4:
                var_idx = random.randint(4, len(A) - 1)
                old_val = A[var_idx]
                
                # Pick a new value
                # To avoid duplicates in A, we pick a random number and check if it's in A
                # Or pick from a pool.
                # Simple: pick random from 1..284. If it's in A (and not old_val), retry or pick another.
                # For speed, just pick random. If it matches another element in A, the swap creates a duplicate element, 
                # which is invalid for a set. 
                # However, our collision logic only looks at sums.
                # If A has duplicate elements, it's not a valid set.
                # We must ensure A remains a set of distinct integers.
                
                new_val = random.randint(1, N)
                if new_val in A and new_val != old_val:
                    # Try a few times to find a unique new_val
                    for _ in range(10):
                        new_val = random.randint(1, N)
                        if new_val not in A or new_val == old_val:
                            break
                    else:
                        continue # Skip this step if we can't find a unique val

                # Calculate Delta E
                # Remove old_val from A (conceptually)
                # Add new_val to A (conceptually)
                
                # We need to update counts.
                # Sums involving old_val: old_val + A[i] for all i.
                # Sums involving new_val: new_val + A[i] for all i.
                
                # Note: The pair (old_val, old_val) is removed.
                # The pair (new_val, new_val) is added.
                
                delta_collisions = 0
                
                # Helper to update counts and return change in collisions
                def update_sum(s, delta):
                    nonlocal delta_collisions
                    old_c = counts.get(s, 0)
                    new_c = old_c + delta
                    if new_c == 0:
                        del counts[s]
                    else:
                        counts[s] = new_c
                    
                    # Collision logic:
                    # If old_c == 1 and new_c == 2: collision created (+1)
                    # If old_c == 2 and new_c == 1: collision removed (-1)
                    # If old_c >= 2 and new_c >= 2: no change in collision count (still 1 conflicting value)
                    # If old_c == 0 and new_c == 1: no change
                    # If old_c == 1 and new_c == 0: no change
                    
                    if old_c == 1 and new_c == 2:
                        delta_collisions += 1
                    elif old_c == 2 and new_c == 1:
                        delta_collisions -= 1

                # Remove old_val contributions
                for i in range(len(A)):
                    if A[i] == old_val and i != var_idx:
                        # Wait, A is a list. A[var_idx] is old_val.
                        # We iterate over all elements.
                        # Sum = old_val + A[i]
                        update_sum(old_val + A[i], -1)
                
                # Also the self-sum 2*old_val
                update_sum(2 * old_val, -1)

                # Add new_val contributions
                # Temporarily replace A[var_idx] with new_val for sum calculation?
                # No, we can just calculate sums with the current A, but replace old_val with new_val in the sum.
                # Actually, the other elements in A are unchanged.
                for i in range(len(A)):
                    # Sum = new_val + A[i]
                    # But if i == var_idx, A[i] is old_val. 
                    # The new self-sum is 2*new_val.
                    # The cross sums are new_val + A[i] for i != var_idx.
                    if i == var_idx:
                        update_sum(2 * new_val, 1)
                    else:
                        update_sum(new_val + A[i], 1)
                
                # Update A
                A[var_idx] = new_val
                
                # Apply delta to global collisions
                collisions += delta_collisions
                
                # Accept/Reject
                if delta_collisions < 0:
                    pass # Accept
                else:
                    # Boltzmann probability
                    if random.random() < pow(2.71828, -delta_collisions / current_temp):
                        pass # Accept
                    else:
                        # Reject: Undo changes
                        # This is expensive. 
                        # Better to copy counts before updating? 
                        # Or just re-calculate if rejected? 
                        # Re-calculation is O(K^2). K=20. 400 ops.
                        # If we reject often, this is slow.
                        
                        # Let's try a different approach: 
                        # Only accept if valid or with prob.
                        # If rejected, we need to revert A and counts.
                        # Reverting counts is complex.
                        
                        # Alternative: 
                        # Keep a backup of counts? No, memory.
                        
                        # Let's just re-calculate from scratch if rejected?
                        # No, that's too slow if we do it every step.
                        
                        # Let's optimize: 
                        # We can just try a different new_val if rejected? No.
                        
                        # Let's use a "Try" structure.
                        # We will implement the update in a way that allows easy rollback?
                        # Or, we can just accept if delta < 0, and for delta > 0, 
                        # we can just skip the update if we don't want to track rollback.
                        # But SA needs to explore.
                        
                        # Let's simplify: 
                        # We will only accept moves that reduce collisions.
                        # This is Hill Climbing, not SA.
                        # Hill Climbing might get stuck.
                        # But with random restarts, it might be enough.
                        
                        # Let's stick to Hill Climbing for simplicity and speed, 
                        # and rely on many restarts.
                        
                        # Revert A
                        A[var_idx] = old_val
                        # Revert counts
                        # We need to undo the updates.
                        # This is tedious.
                        
                        # Let's restart the SA loop if we get stuck?
                        
                        # Actually, let's just use a simple Random Restart Hill Climbing.
                        # 1. Random init.
                        # 2. Local search: find any move that reduces collisions.
                        # 3. If no move, restart.
                        
                        # This is much easier to implement correctly.
                        
                        # Let's abort this SA step and just do Hill Climbing in a separate function.
                        break # Break out of step loop to restart logic?
                        
            current_temp *= cooling_rate
            
        if collisions == 0:
            return A
            
        return None

    # Strategy: Random Restart Hill Climbing
    def hill_climbing(target_size, max_restarts=1000):
        for r in range(max_restarts):
            if time.time() > deadline:
                break
                
            A, counts, collisions = get_initial_state(target_size)
            
            # Local Search
            stuck = False
            while not stuck and time.time() < deadline:
                stuck = True
                # Try all possible swaps for variable elements
                # To make it faster, try random swaps
                for _ in range(50): # Limit steps per restart
                    if time.time() > deadline:
                        break
                        
                    if len(A) <= 4:
                        break
                        
                    var_idx = random.randint(4, len(A) - 1)
                    old_val = A[var_idx]
                    
                    # Try a few random new values
                    for _ in range(5):
                        new_val = random.randint(1, N)
                        if new_val in A and new_val != old_val:
                            continue
                            
                        # Calculate new collisions
                        # We can do this by temporarily updating counts
                        # But to avoid rollback complexity, let's just re-calculate 
                        # if the move looks promising?
                        
                        # Actually, let's just re-calculate the whole thing 
                        # if we find a valid set? No.
                        
                        # Let's use the incremental method but with a backup.
                        # Backup counts is O(1) if we use a dict copy? 
                        # Dict copy is O(S). S is ~400.
                        # 50 steps * 5 tries * 400 ops = 100k ops.
                        # 1000 restarts * 100k = 100M ops. 
                        # Might be too slow in Python.
                        
                        # Let's just use the incremental update and rollback if rejected.
                        
                        # To make rollback easy, we will store the changes.
                        changes = []
                        delta_collisions = 0
                        
                        def update_sum(s, delta):
                            nonlocal delta_collisions
                            old_c = counts.get(s, 0)
                            new_c = old_c + delta
                            changes.append((s, old_c, new_c))
                            
                            if new_c == 0:
                                if s in counts:
                                    del counts[s]
                            else:
                                counts[s] = new_c
                                
                            if old_c == 1 and new_c == 2:
                                delta_collisions += 1
                            elif old_c == 2 and new_c == 1:
                                delta_collisions -= 1

                        # Apply changes
                        for i in range(len(A)):
                            if A[i] == old_val and i != var_idx:
                                update_sum(old_val + A[i], -1)
                        update_sum(2 * old_val, -1)
                        
                        for i in range(len(A)):
                            if i == var_idx:
                                update_sum(2 * new_val, 1)
                            else:
                                update_sum(new_val + A[i], 1)
                                
                        A[var_idx] = new_val
                        collisions += delta_collisions
                        
                        if delta_collisions < 0:
                            stuck = False
                            break # Found an improvement, restart local search loop
                        else:
                            # Rollback
                            for s, old_c, new_c in changes:
                                if new_c == 0:
                                    if s in counts:
                                        del counts[s]
                                else:
                                    counts[s] = new_c
                                # Restore old_c
                                if old_c > 0:
                                    counts[s] = old_c
                                else:
                                    if s in counts:
                                        del counts[s]
                                
                            A[var_idx] = old_val
                            collisions -= delta_collisions
                            
            if collisions == 0:
                return A
                
        return None

    best_set = list(fixed_elements)
    
    # Try to find sets of decreasing size
    for size in range(20, 15, -1):
        result = hill_climbing(size, max_restarts=200)
        if result:
            best_set = result
            break
            
    # If we didn't find a large set, ensure we return the best valid set found
    # The hill_climbing returns None if not found.
    # We should keep track of the best valid set seen.
    
    # Let's modify hill_climbing to return the best valid set found, even if not size 20?
    # No, the goal is to maximize size.
    # If size 20 is not found, we try 19.
    
    # If all fail, return the fixed elements (size 4) or a greedy set.
    # The current program's greedy is decent.
    # Let's add a greedy fallback.
    
    if len(best_set) == 4:
        # Greedy fallback
        a = list(fixed_elements)
        sums = set()
        for i in range(len(a)):
            for j in range(i, len(a)):
                sums.add(a[i] + a[j])
                
        for x in range(12, N + 1):
            valid = True
            for y in a:
                if x + y in sums:
                    valid = False
                    break
            if 2 * x in sums:
                valid = False
            if valid:
                a.append(x)
                for y in a:
                    sums.add(x + y)
                sums.add(2 * x)
                
        best_set = a

    return best_set
