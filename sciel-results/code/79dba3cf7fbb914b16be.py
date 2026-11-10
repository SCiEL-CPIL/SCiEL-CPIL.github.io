import time
import random
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Precompute LCM and targets
    lcm_all = 53542288800
    targets = {i: lcm_all // i for i in range(1, 25)}
    
    # Cache for subset sums of specific sets? 
    # Since sets are small, we can compute on the fly.
    
    def get_subset_sums(vals):
        sums = {0}
        for v in vals:
            new_sums = set()
            for s in sums:
                new_sums.add(s + v)
            sums.update(new_sums)
        return sums

    best_set = []
    best_size = 0
    
    def backtrack(idx, selected, sums_selected):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
            
        # Pruning: even if we add all remaining, we can't beat best
        remaining = 25 - idx # idx is 1-based in this loop structure? 
        # Let's use 0-based index for elements 24..1
        # elements = [24, 23, ..., 1]
        # idx is index in this list.
        if len(selected) + (24 - idx) <= best_size:
            return

        if idx == 24:
            if len(selected) > best_size:
                best_size = len(selected)
                best_set = list(selected)
            return
        
        x = 24 - idx # Current element to consider
        
        # Option 1: Include x
        # Check validity
        valid = True
        if targets[x] in sums_selected:
            valid = False
        
        if valid:
            # Check if adding x breaks any existing y
            for y in selected:
                others = [z for z in selected if z != y]
                sums_others = get_subset_sums([targets[z] for z in others])
                # Check if targets[y] is in sums_others or sums_others + targets[x]
                if targets[y] in sums_others:
                    valid = False
                    break
                if targets[y] - targets[x] in sums_others:
                    valid = False
                    break
        
        if valid:
            new_selected = selected + [x]
            # Update sums
            new_sums = set(sums_selected)
            for s in sums_selected:
                new_sums.add(s + targets[x])
            
            backtrack(idx + 1, new_selected, new_sums)
            
        # Option 2: Exclude x
        backtrack(idx + 1, selected, sums_selected)

    backtrack(0, [], {0})
    
    # Fallback if B&B is too slow or finds nothing (shouldn't happen)
    if not best_set:
        best_set = [24]
        
    return best_set
