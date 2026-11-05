import time
from math import lcm
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin
    
    N = 22
    # Calculate LCM of 1..22
    L = 1
    for i in range(1, N + 1):
        L = lcm(L, i)
        
    # Weights w_i = L // i
    weights = [0] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = L // i
        
    # We want to find a subset S of {1..22} such that all subset sums of {w_i : i in S} are distinct.
    # This is equivalent to S being dissociated in reciprocals.
    
    def is_dissociated(indices):
        if not indices:
            return True
        # Generate all subset sums
        sums = {0}
        for i in indices:
            w = weights[i]
            new_sums = set()
            for s in sums:
                ns = s + w
                new_sums.add(ns)
            # Check for collisions
            if sums & new_sums:
                return False
            sums = sums | new_sums
        return True

    # Backtracking search to find max size
    # We try sizes from 17 down to 1
    # To make it faster, we can use a greedy initialization and then local search
    # Or just backtracking with good ordering
    
    # Order indices by weight ascending (larger i first) to add "small" weights first?
    # Or descending?
    # Let's try descending weight (small i first) as they are more "constraining".
    # If we add constraining ones first, we prune earlier.
    
    indices = list(range(1, N + 1))
    
    best_set = None
    best_size = 0
    
    # Greedy attempt first
    greedy_set = []
    greedy_sums = {0}
    for i in sorted(indices, key=lambda x: weights[x], reverse=True):
        w = weights[i]
        new_sums = set()
        for s in greedy_sums:
            new_sums.add(s + w)
        if not (greedy_sums & new_sums):
            greedy_set.append(i)
            greedy_sums = greedy_sums | new_sums
    best_set = greedy_set
    best_size = len(greedy_set)
    
    # Try to improve with backtracking
    # We want to find a set of size best_size + 1, then best_size + 2, etc.
    # Or just search for max size.
    
    def backtrack(available, current, current_sums, depth):
        nonlocal best_set, best_size
        if time.time() - start_time > deadline:
            return False # Timeout
        
        if len(current) > best_size:
            best_set = current.copy()
            best_size = len(current)
            
        # Pruning: even if we add all remaining, we won't beat best_size
        if len(current) + len(available) <= best_size:
            return True
            
        if not available:
            return True
        
        # Try to add each available element
        # Sort available by weight descending to prune faster?
        # Let's keep original order or sort by weight.
        for idx in range(len(available)):
            elem = available[idx]
            w = weights[elem]
            
            # Check if adding elem causes collision
            new_sums = set()
            for s in current_sums:
                ns = s + w
                new_sums.add(ns)
            
            if current_sums & new_sums:
                continue
                
            # Recurse
            new_current = current + [elem]
            new_sums_union = current_sums | new_sums
            remaining = available[idx + 1:]
            
            # Optimization: if we just want to find ANY set of size K, we can stop early
            # But we want max size.
            
            backtrack(remaining, new_current, new_sums_union, depth + 1)
            
            if time.time() - start_time > deadline:
                return False
                
    # Run backtracking
    # Start with empty set
    # To make it faster, we can try to find a set of size 16, 17...
    # But backtracking explores the tree.
    # We can limit the depth or use a specific target size.
    
    # Let's try to find a set of size 16
    # If found, return it.
    # If not, try 15, etc.
    
    # However, the backtracking above updates best_set.
    # It might be slow if it explores many branches.
    # We can add a target size check.
    
    # Let's just run the backtracking with a time limit.
    # The greedy set is a good lower bound.
    
    backtrack(list(range(1, N + 1)), [], {0}, 0)
    
    if best_set:
        return best_set
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]
