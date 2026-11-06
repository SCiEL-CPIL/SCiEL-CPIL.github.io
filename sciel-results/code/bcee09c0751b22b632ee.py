import time
import random
from math import lcm
from functools import reduce
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Compute LCM of 1..22
    L = reduce(lcm, range(1, 23))
    
    # Weights w[i] = L // (i+1) for i in 0..21
    weights = [L // (i + 1) for i in range(22)]
    
    best_set = [1]
    best_size = 1
    
    def is_dissociated(indices):
        """Check if the subset of indices is dissociated."""
        if not indices:
            return True
        # Use a set of sums
        sums = {0}
        for i in indices:
            w = weights[i]
            new_sums = {s + w for s in sums}
            if sums & new_sums:
                return False
            sums = sums | new_sums
        return True

    def update_best(indices):
        nonlocal best_set, best_size
        if len(indices) > best_size:
            best_size = len(indices)
            best_set = [i + 1 for i in indices]

    # Strategy 1: Backtracking for high cardinalities
    # Try to find a set of size K
    def search_backtracking(K):
        nonlocal best_set, best_size
        # Order indices by weight descending (1, 2, 3...)
        # This helps in finding large sums early?
        # Actually, we just need any valid set.
        indices = list(range(22))
        
        def backtrack(start, current_indices, current_sums):
            nonlocal best_set, best_size
            
            if time.time() > deadline:
                return False
                
            if len(current_indices) == K:
                # Found a valid set
                update_best(current_indices)
                return True
                
            # Pruning: not enough elements left
            if 22 - start < K - len(current_indices):
                return False
                
            # Try to include indices[start]
            i = indices[start]
            w = weights[i]
            new_sums = {s + w for s in current_sums}
            if not (current_sums & new_sums):
                if backtrack(start + 1, current_indices + [i], current_sums | new_sums):
                    return True
                    
            # Try to skip indices[start]
            if backtrack(start + 1, current_indices, current_sums):
                return True
                
            return False

        return backtrack(0, [], {0})

    # Try K from 22 down to 1
    for K in range(22, 0, -1):
        if time.time() > deadline - 5:
            break
        # Run backtracking with a time limit per K?
        # Since backtracking is recursive and doesn't easily yield, 
        # we rely on the global deadline check inside.
        # To prevent one K from eating all time, we can't easily interrupt.
        # So we only try high K's with backtracking.
        if K >= 19:
            search_backtracking(K)
            if best_size >= K:
                return best_set

    # Strategy 2: Randomized Greedy
    # This is faster and can run many times.
    random.seed(42) # Fixed seed for reproducibility in debugging, but random is better?
    # Actually, using random is fine.
    
    attempts = 0
    while time.time() < deadline - 2:
        attempts += 1
        # Random permutation
        perm = list(range(22))
        random.shuffle(perm)
        
        current_indices = []
        current_sums = {0}
        
        for i in perm:
            w = weights[i]
            new_sums = {s + w for s in current_sums}
            if not (current_sums & new_sums):
                current_indices.append(i)
                current_sums = current_sums | new_sums
        
        if len(current_indices) > best_size:
            update_best(current_indices)
            
        # If we found a very large set, we might stop early?
        # No, keep trying to find larger.
        
    return best_set
