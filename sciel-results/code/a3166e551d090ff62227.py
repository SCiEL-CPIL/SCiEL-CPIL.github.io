import time
import math
import random

def construct():
    start_time = time.time()
    timeout = 110  # Safety margin
    
    best_set = [0, 1, 2, 3, 7]
    best_score = 0.0
    
    def calc_score(U):
        if not U:
            return 0.0
        max_u = max(U)
        if max_u == 0:
            return 0.0
        sum_set = set()
        diff_set = set()
        for a in U:
            for b in U:
                sum_set.add(a + b)
                diff_set.add(a - b)
        n_sum = len(sum_set)
        n_diff = len(diff_set)
        if n_sum == 0:
            return 0.0
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        score = 1 + math.log(ratio) / math.log(2 * max_u + 1)
        return score
    
    # Initial evaluation
    best_score = calc_score(best_set)
    
    # Exhaustive search for small N
    # We search subsets of {0, 1, ..., N}
    # We must include 0.
    for N in range(3, 16):
        if time.time() - start_time > timeout:
            break
        # Iterate over subsets of {1, ..., N}
        # Add 0 to each
        range_vals = list(range(1, N + 1))
        n_vals = len(range_vals)
        
        # Use bitmasks
        for mask in range(1, 1 << n_vals):
            if time.time() - start_time > timeout:
                break
            U = [0]
            for i in range(n_vals):
                if mask & (1 << i):
                    U.append(range_vals[i])
            
            # Optional: skip if size is too large or too small?
            # Size 1 is [0], score 0.
            # Size 2 is [0, x].
            
            score = calc_score(U)
            if score > best_score:
                best_score = score
                best_set = U[:]
                
    # Local search / Random search with larger N
    # Try to improve best_set by swapping or adding elements
    # Or just run random search for a bit
    
    # Let's do a simple random search in a larger box
    # e.g., N=30, size up to 8
    random.seed(42)
    iterations = 0
    while time.time() - start_time < timeout - 5:
        iterations += 1
        # Random size 3 to 8
        size = random.randint(3, 8)
        # Random max value
        max_val = random.randint(5, 40)
        # Generate random set containing 0
        # Ensure distinct positive elements
        candidates = list(range(1, max_val + 1))
        random.shuffle(candidates)
        U = [0] + candidates[:size-1]
        random.shuffle(U) # Just in case, though order doesn't matter for sets
        
        score = calc_score(U)
        if score > best_score:
            best_score = score
            best_set = U[:]
            
    return best_set
