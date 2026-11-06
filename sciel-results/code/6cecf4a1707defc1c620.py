import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin: 120s limit, use 115s
    
    # 1. Greedy Baseline
    # We build a set using a greedy approach: add x if it doesn't violate the B2[2] property.
    # We maintain a dictionary 'sum_counts' where key is sum, value is count of representations.
    chosen = []
    sum_counts = {}
    
    for x in range(1, 1001):
        # Check if adding x violates constraints
        # New sums are x + y for y in chosen, and 2*x
        valid = True
        new_sums = []
        for y in chosen:
            s = x + y
            new_sums.append(s)
        new_sums.append(2 * x)
        
        # Check against existing counts
        for s in new_sums:
            current_count = sum_counts.get(s, 0)
            if current_count + 1 > 2:
                valid = False
                break
        
        if valid:
            chosen.append(x)
            # Update counts
            for s in new_sums:
                sum_counts[s] = sum_counts.get(s, 0) + 1

    best_set = list(chosen)
    best_size = len(best_set)
    
    # 2. Local Search
    # We will try to improve the set by:
    # - Adding elements that were previously rejected (if we remove some other element, maybe space opens up)
    # - Swapping elements
    # - Removing elements to allow better additions later
    
    # To speed up local search, we need a fast way to check validity of a candidate set.
    # Since N is small (~400-500), O(N^2) checks are feasible if we do them smartly, 
    # but maintaining state is better.
    
    # However, the stateful update is tricky when removing elements.
    # Let's use a robust check function that is O(N^2) but optimized with early exit,
    # and rely on the fact that local search steps are relatively few compared to pure random search.
    
    def is_valid_b2_2(set_list):
        """Check if set_list is a valid B2[2] set."""
        n = len(set_list)
        if n == 0:
            return True
        
        # Use a dictionary for counts
        counts = {}
        # Optimization: sort helps with early exit? Not really for dict counts.
        # But we can break early if we find a count > 2.
        for i in range(n):
            ai = set_list[i]
            for j in range(i, n):
                s = ai + set_list[j]
                c = counts.get(s, 0) + 1
                if c > 2:
                    return False
                counts[s] = c
        return True

    def get_current_counts(set_list):
        """Return the sum_counts dict for the given set."""
        counts = {}
        n = len(set_list)
        for i in range(n):
            ai = set_list[i]
            for j in range(i, n):
                s = ai + set_list[j]
                counts[s] = counts.get(s, 0) + 1
        return counts

    # Local Search Strategy:
    # We maintain the current set and its counts.
    # Operations:
    # 1. Try to add a random candidate not in the set.
    # 2. Try to swap a random element in the set with a random candidate not in the set.
    # 3. Remove a random element (to diversify).
    
    current_set = list(best_set)
    current_counts = get_current_counts(current_set)
    current_set_set = set(current_set)
    
    # Candidates to try adding: 
    # 1. Values not in current set
    # 2. Prioritize values that are "sparse" in the current sum distribution?
    # For simplicity, we'll pick random values from 1..1000 not in the set.
    
    rng = random.Random(42)
    
    # To make search more effective, we can keep a list of "rejected" candidates from the greedy phase
    # and prioritize trying to add them back after some removals.
    # But for now, random sampling from 1..1000 is a good baseline.
    
    iterations = 0
    no_improvement_counter = 0
    max_no_improvement = 5000
    
    while time.time() < deadline:
        iterations += 1
        
        # Check time frequently
        if iterations % 100 == 0 and time.time() > deadline - 5:
            break
            
        op = rng.random()
        
        if op < 0.6:
            # Try to add a new element
            # Pick a random candidate
            # To improve efficiency, we can pick from a smaller range or use a heuristic.
            # Let's pick from 1..1000
            x = rng.randint(1, 1000)
            if x in current_set_set:
                continue
                
            # Check if adding x is valid
            # New sums: x + y for y in current_set, and 2*x
            valid = True
            new_sums = []
            for y in current_set:
                s = x + y
                new_sums.append(s)
            new_sums.append(2 * x)
            
            for s in new_sums:
                if current_counts.get(s, 0) + 1 > 2:
                    valid = False
                    break
            
            if valid:
                # Add x
                current_set.append(x)
                current_set_set.add(x)
                for s in new_sums:
                    current_counts[s] = current_counts.get(s, 0) + 1
                
                if len(current_set) > best_size:
                    best_set = list(current_set)
                    best_size = len(best_set)
                    no_improvement_counter = 0
                else:
                    no_improvement_counter += 1
                    
        elif op < 0.9:
            # Try to swap
            if not current_set:
                continue
            # Pick an element to remove
            idx_remove = rng.randint(0, len(current_set) - 1)
            y_remove = current_set[idx_remove]
            
            # Pick a candidate to add
            x_add = rng.randint(1, 1000)
            if x_add == y_remove or x_add in current_set_set:
                continue
                
            # Simulate removal of y_remove
            # We need to check validity of (current_set - {y_remove}) + {x_add}
            # Let's compute the counts for the set without y_remove, then check x_add.
            # This is O(N^2) if we recompute, but we can do it smarter?
            # Actually, since N is ~400, O(N^2) is ~160,000 ops. 
            # If we do this for every swap attempt, it might be slow.
            # Let's try a quick check first:
            # If we remove y_remove, some counts decrease.
            # If we add x_add, some counts increase.
            
            # Efficient check:
            # 1. Create a temporary count dict? No, too slow to copy.
            # 2. Calculate the "delta" counts.
            
            # Let's just use the robust O(N^2) check on the modified list.
            # We can optimize by not actually creating the new list until we know it's likely valid?
            # No, we need the list to compute sums.
            
            # Optimization: 
            # Only attempt swap if the "potential" is high.
            # For now, let's do the full check but limit the number of swap attempts per second?
            # Or just let it run. 160k ops is fast in Python (~0.01s).
            # If we do 1000 swaps/sec, that's 160M ops/sec, which is too slow for pure Python.
            # So we need a faster validity check or fewer checks.
            
            # Let's use the incremental update approach for swaps too?
            # It's complex. Let's stick to the incremental add/remove for the "Add" operation (which is O(N)).
            # For "Swap", we can simulate:
            # 1. Remove y_remove: update counts (O(N))
            # 2. Check if x_add is valid in the new state (O(N))
            # 3. If valid, add x_add (O(N))
            # 4. If invalid, revert (O(N))
            
            # This makes Swap O(N) as well!
            
            # Step 1: Remove y_remove
            removal_sums = []
            for y in current_set:
                if y != y_remove:
                    removal_sums.append(y_remove + y)
            removal_sums.append(2 * y_remove)
            
            # Apply removal to current_counts
            for s in removal_sums:
                current_counts[s] -= 1
                if current_counts[s] == 0:
                    # Optional: del current_counts[s] to keep dict small, but not strictly necessary
                    pass
            
            # Remove from list and set
            current_set.pop(idx_remove)
            current_set_set.remove(y_remove)
            
            # Step 2: Check x_add
            valid_add = True
            add_sums = []
            for y in current_set:
                s = x_add + y
                add_sums.append(s)
            add_sums.append(2 * x_add)
            
            for s in add_sums:
                if current_counts.get(s, 0) + 1 > 2:
                    valid_add = False
                    break
            
            if valid_add:
                # Step 3: Add x_add
                for s in add_sums:
                    current_counts[s] = current_counts.get(s, 0) + 1
                current_set.append(x_add)
                current_set_set.add(x_add)
                
                if len(current_set) > best_size:
                    best_set = list(current_set)
                    best_size = len(best_set)
                    no_improvement_counter = 0
                else:
                    no_improvement_counter += 1
            else:
                # Step 4: Revert removal of y_remove
                # Add y_remove back to counts
                for s in removal_sums:
                    current_counts[s] = current_counts.get(s, 0) + 1
                
                # Add y_remove back to list and set
                # We don't care about order for correctness, but let's keep it somewhat sorted or just append
                current_set.append(y_remove)
                current_set_set.add(y_remove)
                
                # Note: The list order might be jumbled, but that doesn't affect validity.
                # However, for future "Add" operations, we iterate over current_set.
                # So it's fine.
                
                no_improvement_counter += 1
                
        else:
            # Remove a random element to escape local optima
            if not current_set:
                continue
            idx_remove = rng.randint(0, len(current_set) - 1)
            y_remove = current_set[idx_remove]
            
            # Remove y_remove from counts
            removal_sums = []
            for y in current_set:
                if y != y_remove:
                    removal_sums.append(y_remove + y)
            removal_sums.append(2 * y_remove)
            
            for s in removal_sums:
                current_counts[s] -= 1
            
            current_set.pop(idx_remove)
            current_set_set.remove(y_remove)
            no_improvement_counter += 1
        
        # If we've had a lot of no-improvement steps, we might want to restart or change strategy
        # But the random nature should handle it.
        
    return best_set

# EVOLVE-BLOCK-END
