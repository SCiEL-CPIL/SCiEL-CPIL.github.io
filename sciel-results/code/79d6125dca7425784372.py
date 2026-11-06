import time
import random
from itertools import combinations

# LCM of 1..22
_LCM = 232792560
_WEIGHTS = {x: _LCM // x for x in range(1, 23)}

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 60s limit
    
    all_elements = list(range(1, 23))
    # Sort elements by weight descending (i.e., x ascending)
    # Larger weights are "safer" to add early as they create larger gaps
    # This heuristic is a guess, but often works for packing problems.
    sorted_elements = sorted(all_elements, key=lambda x: -_WEIGHTS[x])
    
    best_set = set()
    best_size = 0
    
    # Initial quick greedy to get a baseline
    # Try a few random orders to find a decent starting point
    for _ in range(20):
        random.shuffle(sorted_elements)
        current_sums = {0}
        current_set = set()
        for x in sorted_elements:
            w = _WEIGHTS[x]
            new_sums = {s + w for s in current_sums}
            if not new_sums & current_sums:
                current_sums |= new_sums
                current_set.add(x)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
    if best_size == 0:
        return [1]
        
    # Backtracking search to find larger sets
    # We use a recursive function with a time check
    def backtrack(current_sums, current_set, available):
        nonlocal best_set, best_size
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
        # Pruning: if we can't beat best_size even with all remaining elements
        if len(current_set) + len(available) <= best_size:
            return
            
        # Time check
        if time.time() > deadline:
            return
            
        # Try adding each available element
        # Heuristic: try elements with larger weights first?
        # Or just iterate through the sorted list
        for i in range(len(available)):
            x = available[i]
            w = _WEIGHTS[x]
            
            # Check collision
            # new_sums = {s + w for s in current_sums}
            # Collision if any s + w is in current_sums
            # This is equivalent to checking if (current_sums + w) intersects current_sums
            # Or simply: if any (s1 - s2) == w for s1, s2 in current_sums?
            # No, that's for difference.
            # We need to check if (s + w) in current_sums for any s in current_sums.
            
            # Optimization: 
            # If w is large, it's less likely to collide.
            # We can just generate the set and check intersection.
            # For k=12, len(current_sums) = 4096. 
            # Generating 4096 new ints and checking intersection with 4096 ints is fast.
            
            new_sums = set()
            collision = False
            for s in current_sums:
                ns = s + w
                if ns in current_sums:
                    collision = True
                    break
                new_sums.add(ns)
                
            if not collision:
                # Recurse
                new_available = available[:i] + available[i+1:]
                backtrack(current_sums | new_sums, current_set | {x}, new_available)
                
                if time.time() > deadline:
                    return
                
    # Start backtracking from the best greedy solution found so far?
    # Or start from scratch?
    # Starting from scratch with backtracking is better as it explores the whole space.
    # But we can use the best greedy set as a "seed" for the initial state?
    # No, backtracking from empty set is standard.
    # However, to speed up, we can start the backtracking with the elements already in best_set?
    # That's risky if the greedy set is not a subset of the optimal set.
    # Let's just run backtracking from empty set, but use the best_size to prune.
    
    # To make backtracking faster, we can pass the remaining elements in a specific order.
    # Let's try to force the inclusion of the "easiest" elements (largest weights) first?
    # Actually, the `available` list is passed down. 
    # We can sort `available` at each step? That's expensive.
    # Let's just iterate through the global sorted list and check if the element is in `available`.
    
    # Refined Backtracking:
    # Instead of passing `available` list, pass an index or a bitmask?
    # Bitmask is efficient.
    
    def backtrack_mask(current_sums, current_mask, current_size):
        nonlocal best_set, best_size
        
        if current_size > best_size:
            best_size = current_size
            best_set = set(i for i in range(1, 23) if (current_mask >> (i-1)) & 1)
            
        if current_size + (22 - current_size) <= best_size: # Trivial bound
            return
            
        if time.time() > deadline:
            return
            
        # Try adding any element not in current_mask
        # We iterate through sorted_elements (largest weight first)
        for x in sorted_elements:
            bit = 1 << (x - 1)
            if current_mask & bit:
                continue
                
            w = _WEIGHTS[x]
            
            # Check collision
            collision = False
            new_sums = set()
            for s in current_sums:
                ns = s + w
                if ns in current_sums:
                    collision = True
                    break
                new_sums.add(ns)
                
            if not collision:
                backtrack_mask(current_sums | new_sums, current_mask | bit, current_size + 1)
                
                if time.time() > deadline:
                    return
                
    # Start with empty set
    backtrack_mask({0}, 0, 0)
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
