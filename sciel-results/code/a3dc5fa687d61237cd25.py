import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    N = 1000
    # Max sum is 2000. We need indices 0..2000.
    max_sum = 2 * N
    
    def is_valid_add_np(counts, x, chosen_arr):
        """
        Check if x can be added.
        counts: np.array of size max_sum+1
        x: int
        chosen_arr: np.array of chosen elements
        """
        if len(chosen_arr) == 0:
            return True
        
        # Sums x + y for y in chosen
        # x is int, chosen_arr is int array
        sums = x + chosen_arr
        
        # Check if any count is >= 2
        # We need to check counts[s] for s in sums.
        # Note: x+x is included in sums if x is in chosen, but x is not in chosen yet.
        # Wait, the condition is a <= b.
        # If we add x, the new pairs are (x, x) and (x, y) for y in chosen.
        # (x, x) sum is 2x.
        # (x, y) sum is x+y.
        
        # Check 2x
        if counts[2 * x] >= 2:
            return False
            
        # Check x + y for y in chosen
        # We only need to check y >= x? No, the set is unordered in storage, 
        # but the pair (a,b) with a<=b is unique.
        # If we maintain chosen sorted, we can just check y >= x?
        # Actually, if y < x, the pair is (y, x) which is the same as (x, y).
        # The count for sum s includes all pairs (a,b) in S such that a+b=s.
        # If we add x, we add pairs (x, x) and (x, y) for all y in S.
        # So we just need to check if counts[s] + 1 > 2 for any s in {2x} U {x+y | y in S}.
        # i.e. counts[s] >= 2.
        
        current_counts = counts[sums]
        if np.any(current_counts >= 2):
            return False
            
        return True

    def add_element_np(counts, x, chosen_list, chosen_set):
        """Add x to the set and update counts."""
        # Update counts
        # 2x
        counts[2 * x] += 1
        
        # x + y for y in chosen_list
        if chosen_list:
            chosen_arr = np.array(chosen_list)
            sums = x + chosen_arr
            # Use np.add.at to handle potential duplicate sums (though sums are distinct if chosen is distinct? 
            # No, x+y1 could equal x+y2 only if y1=y2. Since y are distinct, sums are distinct.
            # So simple indexing works? 
            # counts[sums] += 1 works if sums are unique. They are unique because y are unique.
            counts[sums] += 1
            
        chosen_list.append(x)
        chosen_set.add(x)
        
    def remove_element_np(counts, x, chosen_list, chosen_set):
        """Remove x from the set and update counts."""
        counts[2 * x] -= 1
        
        if x in chosen_set:
            chosen_set.remove(x)
        else:
            return False
            
        # Remove x from list
        # Find index
        try:
            idx = chosen_list.index(x)
            chosen_list.pop(idx)
        except ValueError:
            pass
            
        # Update counts
        if chosen_list:
            chosen_arr = np.array(chosen_list)
            sums = x + chosen_arr
            counts[sums] -= 1
            
        return True

    best_size = 0
    best_chosen_list = []
    best_chosen_set = set()
    
    # Helper to update best
    def update_best(chosen_list, chosen_set):
        nonlocal best_size, best_chosen_list, best_chosen_set
        if len(chosen_list) > best_size:
            best_size = len(chosen_list)
            best_chosen_list = chosen_list.copy()
            best_chosen_set = chosen_set.copy()

    # Phase 1: Multiple greedy passes
    # We will run several greedy passes with different orders.
    
    # 1. Forward greedy
    if time.time() < deadline:
        counts = np.zeros(max_sum + 1, dtype=np.int32)
        chosen_list = []
        chosen_set = set()
        
        for x in range(1, N + 1):
            if is_valid_add_np(counts, x, np.array(chosen_list) if chosen_list else np.array([])):
                add_element_np(counts, x, chosen_list, chosen_set)
        
        update_best(chosen_list, chosen_set)

    # 2. Backward greedy
    if time.time() < deadline - 5:
        counts = np.zeros(max_sum + 1, dtype=np.int32)
        chosen_list = []
        chosen_set = set()
        
        for x in range(N, 0, -1):
            if is_valid_add_np(counts, x, np.array(chosen_list) if chosen_list else np.array([])):
                add_element_np(counts, x, chosen_list, chosen_set)
        
        update_best(chosen_list, chosen_set)

    # 3. Random greedy passes
    random.seed(42)
    num_random_passes = 50 # Adjust based on speed
    
    for i in range(num_random_passes):
        if time.time() > deadline - 20:
            break
            
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        counts = np.zeros(max_sum + 1, dtype=np.int32)
        chosen_list = []
        chosen_set = set()
        
        # Optimization: Pre-compute array for empty set? No, it grows.
        # Converting list to array every step is O(N). 
        # We can maintain a numpy array and a mask, or just accept the conversion cost if N is small enough.
        # N=1000, conversion is fast.
        
        for x in order:
            if is_valid_add_np(counts, x, np.array(chosen_list) if chosen_list else np.array([])):
                add_element_np(counts, x, chosen_list, chosen_set)
                
        update_best(chosen_list, chosen_set)

    # Phase 2: Local Search (Swaps)
    # Start from the best found so far
    if time.time() < deadline - 5:
        # Reconstruct counts for best
        counts = np.zeros(max_sum + 1, dtype=np.int32)
        current_list = best_chosen_list.copy()
        current_set = best_chosen_set.copy()
        
        # Build counts from scratch for safety
        for x in current_list:
             # This is O(N^2). N~100-200. 200^2 = 40000. Fast.
             if current_set:
                 # We can just use the add logic but we need to be careful not to double count if we were building incrementally.
                 # Since we are starting from scratch, we can just loop.
                 pass
        
        # Let's just rebuild counts properly
        counts = np.zeros(max_sum + 1, dtype=np.int32)
        for i in range(len(current_list)):
            a = current_list[i]
            counts[2*a] += 1
            for j in range(i+1, len(current_list)):
                b = current_list[j]
                counts[a+b] += 1
                # Note: add_element_np adds x+x and x+y for ALL y in list.
                # If we build by iterating i, j, we count each pair once.
                # My add_element_np assumes x is NEW.
                # So I can't use add_element_np to build from scratch easily without removing the "x+x" logic if x is already there?
                # No, add_element_np adds x+x and x+y for y in existing.
                # If I add elements one by one, it works.
                
        # Let's re-initialize and add one by one to use the existing functions
        counts = np.zeros(max_sum + 1, dtype=np.int32)
        temp_list = []
        temp_set = set()
        for x in current_list:
             add_element_np(counts, x, temp_list, temp_set)
             
        # Now counts are correct for current_list.
        # temp_list is same as current_list.
        
        improved = True
        no_improve_count = 0
        max_no_improve = 10
        
        while improved and time.time() < deadline - 2:
            improved = False
            if no_improve_count >= max_no_improve:
                break
                
            # Try random swaps
            # Pick a random element to remove
            if not current_list:
                break
                
            old_x = random.choice(current_list)
            
            # Remove it
            remove_element_np(counts, old_x, current_list, current_set)
            
            # Try to add a different element
            found = False
            # Candidates: all 1..N not in current_set
            # To speed up, we can check a random subset of candidates
            candidates = [x for x in range(1, N + 1) if x not in current_set]
            random.shuffle(candidates)
            
            # Try first 50 candidates
            for x in candidates[:50]:
                # Check validity
                if is_valid_add_np(counts, x, np.array(current_list) if current_list else np.array([])):
                    add_element_np(counts, x, current_list, current_set)
                    found = True
                    improved = True
                    break
                    
            if not found:
                # Revert: put old_x back
                add_element_np(counts, old_x, current_list, current_set)
                no_improve_count += 1
            else:
                no_improve_count = 0
                update_best(current_list, current_set)

    return best_chosen_list
