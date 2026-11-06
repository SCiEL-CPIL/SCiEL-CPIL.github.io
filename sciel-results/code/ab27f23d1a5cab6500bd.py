import time
import sys

# Global variables
best_set = []
best_size = 0
start_time = 0
time_limit = 115.0

# Max possible sum is 284 + 284 = 568
# We need an array of size 569 (indices 0 to 568)
SUMS_SIZE = 570

# Precompute bits for sums
sum_bits = [1 << i for i in range(SUMS_SIZE)]

def construct():
    global best_set, best_size, start_time
    
    start_time = time.time()
    
    # Initial prescribed set
    initial_set = [1, 2, 5, 11]
    initial_mask = 0
    
    # Mark initial sums
    for i in range(len(initial_set)):
        for j in range(i, len(initial_set)):
            s = initial_set[i] + initial_set[j]
            initial_mask |= sum_bits[s]
            
    best_set = initial_set[:]
    best_size = len(initial_set)
    
    # Candidates are 12 to 284
    # We filter out candidates that conflict with the initial set immediately
    candidates = []
    for x in range(12, 285):
        valid = True
        # Check 2*x
        if initial_mask & sum_bits[2 * x]:
            valid = False
        # Check x + y for y in initial_set
        if valid:
            for y in initial_set:
                s = x + y
                if initial_mask & sum_bits[s]:
                    valid = False
                    break
        if valid:
            candidates.append(x)
            
    # Sort candidates. 
    # Heuristic: Descending order often works well for Golomb rulers to spread out marks.
    candidates.sort(reverse=True)
    
    # We will use a recursive DFS with pruning.
    
    def dfs(current_set, mask, start_idx, target_size):
        global best_set, best_size
        
        if time.time() - start_time > time_limit:
            return False
            
        current_size = len(current_set)
        
        if current_size > best_size:
            best_size = current_size
            best_set = current_set[:]
            
        if current_size == target_size:
            return True
            
        # Pruning: If we can't reach target_size with remaining candidates
        if current_size + (len(candidates) - start_idx) < target_size:
            return False
            
        # Iterate through candidates from start_idx
        for i in range(start_idx, len(candidates)):
            x = candidates[i]
            
            # Check if x can be added
            # 1. 2*x
            if mask & sum_bits[2 * x]:
                continue
                
            # 2. x + y for y in current_set
            can_add = True
            new_mask = mask
            for y in current_set:
                s = x + y
                if new_mask & sum_bits[s]:
                    can_add = False
                    break
                new_mask |= sum_bits[s]
                
            if not can_add:
                continue
                
            # Add 2*x to new_mask
            new_mask |= sum_bits[2 * x]
            
            # Recurse
            current_set.append(x)
            if dfs(current_set, new_mask, i + 1, target_size):
                return True
            current_set.pop()
                
        return False

    # Try sizes from 20 down to best_size + 1
    # We know best_size is at least 4.
    # Max size is 20.
    
    # To avoid spending too much time on an impossible target 20, 
    # we can add a time slice for each target.
    # Or just run until time limit.
    # Given the speedup from bitmask, we might solve 20 quickly if it exists.
    # If it doesn't, we might waste time.
    # Let's try target 20 first. If it takes > 10s, break and try 19?
    # The evaluator kills at 150s. Our limit is 115s.
    
    for target in range(20, best_size, -1):
        if time.time() - start_time > time_limit - 5:
            break
            
        # Reset state
        init_set_copy = initial_set[:]
        
        if dfs(init_set_copy, initial_mask, 0, target):
            return best_set
            
    return best_set
