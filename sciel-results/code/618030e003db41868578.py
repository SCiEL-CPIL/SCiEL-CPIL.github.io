import math
import time
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s deadline, 5s safety margin for cleanup/return

    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def lcm(a, b):
        return a * b // gcd(a, b)

    L = reduce(lcm, range(1, 25))
    
    # Create list of (value, x) sorted by value ascending
    items = []
    for x in range(1, 25):
        items.append((L // x, x))
    
    # Sort by value ascending (smallest reciprocals first = largest x first)
    items.sort(key=lambda p: p[0])
    
    best_result = [1]  # at minimum, return {1}
    best_size = 1
    
    # sums will be a set of integers
    # We use in-place updates with a list of added elements to undo
    
    def backtrack(idx, selected, sums):
        nonlocal best_result, best_size
        
        if time.time() > deadline:
            return
        
        # Pruning: if we can't beat best_size, return
        # Max possible size is len(items) - idx + len(selected)
        if len(selected) + (len(items) - idx) <= best_size:
            return
            
        if len(selected) > best_size:
            best_size = len(selected)
            best_result = list(selected)
        
        # Try to add each remaining element
        for i in range(idx, len(items)):
            if time.time() > deadline:
                return
            
            v, x = items[i]
            
            # Check if v is already formable by a subset of selected elements
            if v in sums:
                continue
            
            # If not, we can try adding x
            selected.append(x)
            
            # Update sums in-place
            # Calculate new sums to add
            # We iterate over a copy of the current sums to avoid modification during iteration issues
            # But we only add new ones.
            # Optimization: if sum of all current sums < v, we don't need to check? 
            # No, we just need to check if v is in sums.
            
            # To update sums: new_sums = {s + v for s in sums}
            # We need to add these to sums.
            # We must remember which ones we added to undo them.
            
            # Efficient way:
            # 1. Get current sums
            # 2. Generate candidate new sums
            # 3. Filter out those already in sums
            # 4. Add filtered ones to sums
            # 5. Store the list of added ones for undo
            
            current_sums_list = list(sums)
            to_add = []
            for s in current_sums_list:
                ns = s + v
                if ns not in sums:
                    to_add.append(ns)
            
            sums.update(to_add)
            
            backtrack(i + 1, selected, sums)
            
            # Undo
            selected.pop()
            sums.difference_update(to_add)
            
            if time.time() > deadline:
                return
    
    backtrack(0, [], {0})
    
    return best_result
