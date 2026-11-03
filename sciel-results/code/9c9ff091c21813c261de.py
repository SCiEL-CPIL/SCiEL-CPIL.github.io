import numpy as np
import random
import itertools

def score_set(U):
    U = list(set(U))
    if 0 not in U:
        return -1
    U.sort()
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # Calculate sumset
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
    # Start with a promising seed
    best_set = [0, 1, 3, 8, 16]
    best_score = score_set(best_set)
    
    # Local search
    # We allow adding, removing, or replacing elements
    # Max size to consider
    max_size = 10
    min_size = 3
    
    current_set = list(best_set)
    current_score = best_score
    
    # Randomization
    random.seed(42)
    
    # Define a range for new values
    # Based on max_u, we can try values up to 2*max_u or so
    # But to keep log_max small, we prefer small values.
    
    iterations = 5000
    
    for _ in range(iterations):
        # Pick a move
        move_type = random.choice(['replace', 'add', 'remove'])
        
        new_set = list(current_set)
        
        if move_type == 'replace' and len(new_set) > 0:
            # Replace a random non-zero element
            if len(new_set) > 1:
                idx = random.randint(1, len(new_set)-1) # Don't replace 0
                old_val = new_set[idx]
                # Try values in a range around old_val or globally
                # Range: 1 to 2*max(current)
                max_val = max(new_set)
                upper_bound = max(20, 2 * max_val)
                new_val = random.randint(1, upper_bound)
                if new_val not in new_set:
                    new_set[idx] = new_val
                    new_set.sort()
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current_score = new_score
                        current_set = new_set
                        if current_score > best_score:
                            best_score = current_score
                            best_set = list(current_set)
        
        elif move_type == 'add' and len(new_set) < max_size:
            max_val = max(new_set)
            upper_bound = max(20, 2 * max_val)
            new_val = random.randint(1, upper_bound)
            if new_val not in new_set:
                new_set.append(new_val)
                new_set.sort()
                new_score = score_set(new_set)
                if new_score > current_score:
                    current_score = new_score
                    current_set = new_set
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(current_set)
        
        elif move_type == 'remove' and len(new_set) > min_size:
            # Remove a random non-zero element
            idx = random.randint(1, len(new_set)-1)
            new_set.pop(idx)
            new_score = score_set(new_set)
            if new_score > current_score:
                current_score = new_score
                current_set = new_set
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(current_set)
            
            # Occasionally allow worsening moves (simulated annealing style) 
            # to escape local minima, but keep it simple for now.
            # If we want to be more robust:
            # if new_score > current_score - 0.01: ...

    return best_set
