import time
import random

def is_sidon(subset):
    sums = set()
    for i in range(len(subset)):
        for j in range(i, len(subset)):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def has_sidon_4(A):
    # Check if any 4-element subset is Sidon
    from itertools import combinations
    for comb in combinations(A, 4):
        if is_sidon(list(comb)):
            return True
    return False

def max_sidon_size(A):
    # Assume max is 3 or 4
    if has_sidon_4(A):
        return 4
    else:
        return 3

def construct():
    start_time = time.time()
    deadline = start_time + 50 # Safety margin
    
    # Initial set: dense interval
    A = list(range(12))
    best_A = A[:]
    best_size = max_sidon_size(best_A)
    
    # Local Search
    while time.time() < deadline:
        # Randomly pick an index to replace
        idx = random.randint(0, 11)
        old_val = A[idx]
        
        # Pick a new value
        new_val = random.randint(0, 128)
        if new_val in A:
            continue
            
        A[idx] = new_val
        
        current_size = max_sidon_size(A)
        
        if current_size < best_size:
            best_size = current_size
            best_A = A[:]
            if best_size == 3:
                break # Can't get better than 3
        else:
            # Revert
            A[idx] = old_val
            
    return best_A
