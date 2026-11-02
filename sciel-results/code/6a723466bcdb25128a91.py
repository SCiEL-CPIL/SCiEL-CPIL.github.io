import numpy as np
import random
import time

def score_set(U):
    U = list(set(U))
    if 0 not in U:
        return -1
    U.sort()
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # Calculate sumset and diffset
    # Using sets for O(1) lookups and addition
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    
    sum_size = len(sumset)
    diff_size = len(diffset)
    
    if sum_size == 0:
        return -1
        
    ratio = diff_size / sum_size
    if ratio <= 0:
        return -1
        
    log_ratio = np.log2(ratio)
    log_max = np.log2(2 * max_u + 1)
    
    if log_max == 0:
        return -1
        
    return 1 + log_ratio / log_max

def construct():
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    best_set = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # Initial guess
    best_score = score_set(best_set)
    
    # Hill Climbing parameters
    max_iterations_per_restart = 5000
    max_value_range = 500 # Initial max value for random generation
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize random set
        # Start with a small size and grow
        current_size = random.randint(5, 15)
        current_max_val = random.randint(50, 200)
        
        # Generate initial set
        U = {0}
        while len(U) < current_size:
            val = random.randint(1, current_max_val)
            U.add(val)
        
        current_set = list(U)
        current_score = score_set(current_set)
        
        # Local search
        for i in range(max_iterations_per_restart):
            if time.time() - start_time > time_limit:
                break
                
            # Generate a mutation
            # 1. Add a new element
            # 2. Remove an element (if size > 1)
            # 3. Change an element
            
            mutation_type = random.choice(['add', 'remove', 'change'])
            new_set = list(current_set)
            
            if mutation_type == 'add':
                # Try adding a value slightly larger than current max or random
                if random.random() < 0.5:
                    new_val = random.randint(1, int(max(current_set) * 1.5) + 10)
                else:
                    new_val = random.randint(1, max_value_range)
                
                if new_val not in new_set:
                    new_set.append(new_val)
                else:
                    continue
                    
            elif mutation_type == 'remove':
                if len(new_set) > 1:
                    # Remove a random element, keep 0
                    non_zero = [x for x in new_set if x != 0]
                    if non_zero:
                        idx = random.randint(0, len(non_zero) - 1)
                        val_to_remove = non_zero[idx]
                        new_set.remove(val_to_remove)
                    else:
                        continue
                else:
                    continue
                    
            else: # change
                if len(new_set) > 1:
                    non_zero = [x for x in new_set if x != 0]
                    if non_zero:
                        idx = random.randint(0, len(non_zero) - 1)
                        old_val = non_zero[idx]
                        # Replace with a new random value
                        if random.random() < 0.5:
                            new_val = random.randint(1, int(max(new_set) * 1.2) + 10)
                        else:
                            new_val = random.randint(1, max_value_range)
                        
                        if new_val not in new_set or new_val == old_val:
                            new_set[new_set.index(old_val)] = new_val
                        else:
                            continue
                    else:
                        continue
                else:
                    continue

            # Ensure 0 is still in the set (it should be, as we don't remove/replace it)
            if 0 not in new_set:
                continue
            
            new_score = score_set(new_set)
            
            if new_score > current_score:
                current_score = new_score
                current_set = new_set
                
                if current_score > best_score:
                    best_score = current_score
                    best_set = current_set
                    
                # Optional: expand max_value_range if we are hitting limits
                # max_value_range = max(max_value_range, int(max(current_set) * 1.2))
            else:
                # Accept worse moves with low probability to escape local minima (Simulated Annealing style)
                # But simple Hill Climbing is often sufficient for these smooth landscapes initially.
                # Let's add a small probability of accepting worse moves if we've been stuck.
                pass # Skipping SA for simplicity first, Hill Climbing is robust here

    return best_set
