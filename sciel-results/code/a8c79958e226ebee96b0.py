import numpy as np
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
    best_score = -1
    best_set = [0, 1, 3, 7]
    
    # Update best with initial guess
    current_best_score = score_set(best_set)
    if current_best_score > best_score:
        best_score = current_best_score

    # Exhaustive search for small sets
    # We search for subsets of {1, ..., max_val} of size k
    # U = {0} + subset
    
    max_val_search = 25
    
    for n in range(3, 7): # Size of U: 3 to 6
        k = n - 1 # Number of elements to choose from 1..max_val_search
        if k > max_val_search:
            continue
            
        # To speed up, we can limit the max value for larger n if needed, 
        # but 25 is small enough.
        
        for comb in itertools.combinations(range(1, max_val_search + 1), k):
            U = [0] + list(comb)
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U

    return best_set
