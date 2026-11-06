import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 256
    
    def is_valid_set(S):
        S_list = list(S)
        S_list.sort()
        for i, a in enumerate(S_list):
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                for k in range(j, len(S_list)):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def can_add(x, current_set):
        # current_set contains elements > x
        # Check if x divides b+c for any b,c in current_set with b<=c
        # current_set is a set, convert to list for iteration
        # Optimization: If current_set is large, this is slow.
        # We can precompute sums? No, x changes.
        c_list = list(current_set)
        L = len(c_list)
        for i in range(L):
            b = c_list[i]
            for k in range(i, L):
                c = c_list[k]
                if (b + c) % x == 0:
                    return False
        return True

    best_size = 0
    best_set = set()
    
    # Initialize with a strong greedy solution to help pruning
    # Greedy from largest to smallest
    greedy_set = []
    for x in range(n, 0, -1):
        # Check if x can be added to greedy_set
        # greedy_set contains elements > x
        valid = True
        L = len(greedy_set)
        for i in range(L):
            b = greedy_set[i]
            for k in range(i, L):
                c = greedy_set[k]
                if (b + c) % x == 0:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            greedy_set.append(x)
    
    if len(greedy_set) > best_size:
        best_size = len(greedy_set)
        best_set = set(greedy_set)
        
    candidates = list(range(n, 0, -1))
    
    def branch_and_bound(idx, current_set):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
            
        # Pruning
        # Upper bound: current size + all remaining candidates
        remaining = n - idx
        if len(current_set) + remaining <= best_size:
            return
            
        if idx == n:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
            return
            
        x = candidates[idx]
        
        # Option 1: Include x
        if can_add(x, current_set):
            current_set.add(x)
            branch_and_bound(idx + 1, current_set)
            current_set.discard(x)
            
        # Option 2: Exclude x
        branch_and_bound(idx + 1, current_set)

    # Run B&B
    branch_and_bound(0, set())
    
    return list(best_set)

# EVOLVE-BLOCK-END
