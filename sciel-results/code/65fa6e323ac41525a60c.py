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
    # Use a local search / hill climbing approach
    # Start with a few diverse candidates
    candidates = [
        [0, 1, 3],
        [0, 1, 3, 7],
        [0, 1, 2, 4],
        [0, 1, 4, 6],
        [0, 2, 3, 5],
        [0, 1, 3, 4, 8],
        [0, 1, 2, 5, 9]
    ]
    
    # Add some random starts
    random.seed(42)
    for _ in range(50):
        size = random.randint(3, 12)
        max_val = random.randint(10, 100)
        # Ensure 0 is included
        others = random.sample(range(1, max_val + 1), size - 1)
        candidates.append([0] + others)
    
    best_set = max(candidates, key=score_set)
    best_score = score_set(best_set)
    
    # Hill Climbing
    max_iterations = 5000
    start_time = time.time()
    
    for _ in range(max_iterations):
        if time.time() - start_time > 100:  # Leave time for scoring
            break
            
        current_set = best_set
        current_score = best_score
        
        # Generate neighbors
        # 1. Add a random element
        # 2. Remove a random element (not 0)
        # 3. Shift a random element (not 0)
        
        # Try adding
        max_val = max(current_set) if current_set else 0
        # Limit max_val to keep denominator small
        max_search_val = max(50, max_val * 2)
        
        # Random addition
        if random.random() < 0.3:
            new_val = random.randint(1, max_search_val)
            if new_val not in current_set:
                new_set = current_set + [new_val]
                s = score_set(new_set)
                if s > best_score:
                    best_score = s
                    best_set = new_set
        # Random removal
        elif random.random() < 0.3:
            if len(current_set) > 1:
                # Remove a non-zero element
                rem_idx = random.randint(1, len(current_set) - 1)
                new_set = current_set[:rem_idx] + current_set[rem_idx+1:]
                s = score_set(new_set)
                if s > best_score:
                    best_score = s
                    best_set = new_set
        # Random shift
        else:
            if len(current_set) > 1:
                shift_idx = random.randint(1, len(current_set) - 1)
                old_val = current_set[shift_idx]
                new_val = max(1, old_val + random.randint(-10, 10))
                if new_val not in current_set:
                    new_set = current_set.copy()
                    new_set[shift_idx] = new_val
                    new_set.sort()
                    s = score_set(new_set)
                    if s > best_score:
                        best_score = s
                        best_set = new_set
                        
    return best_set
