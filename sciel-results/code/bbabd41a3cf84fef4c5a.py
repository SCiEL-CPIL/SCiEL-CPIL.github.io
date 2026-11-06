import math
import time
import random

def construct():
    start_time = time.time()
    deadline = 85  # safety margin before 90s kill
    
    n = 256
    
    # Perfect squares up to 2*256 = 512
    max_sum = 2 * n
    squares = set()
    i = 1
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1
    
    # Step 1: Remove numbers where 2*a is a perfect square (diagonal constraint)
    candidates = []
    for a in range(1, n + 1):
        if 2 * a not in squares:
            candidates.append(a)
    
    # Build conflict sets for fast lookup
    # Using bitsets or sets? Sets are fine for N=200.
    conflict = {a: set() for a in candidates}
    cand_set = set(candidates)
    for idx, a in enumerate(candidates):
        for b in candidates[idx+1:]:
            if a + b in squares:
                conflict[a].add(b)
                conflict[b].add(a)
                
    def is_valid_set(s):
        s_set = set(s)
        for a in s:
            for b in s:
                if a <= b and (a+b) in squares:
                    return False
        return True

    def greedy_select(order):
        selected = []
        selected_set = set()
        for a in order:
            if a in selected_set:
                continue
            # Check if a conflicts with any already selected
            # Optimization: check intersection of conflict[a] and selected_set
            if not (conflict[a] & selected_set):
                selected.append(a)
                selected_set.add(a)
        return selected

    best_set = set()
    
    # 1. Multiple Greedy Strategies
    strategies = []
    
    # Natural
    strategies.append(candidates)
    # Reverse
    strategies.append(list(reversed(candidates)))
    # Degree Asc
    strategies.append(sorted(candidates, key=lambda x: len(conflict[x])))
    # Degree Desc
    strategies.append(sorted(candidates, key=lambda x: -len(conflict[x])))
    
    # Random shuffles
    for _ in range(20):
        shuffled = candidates[:]
        random.shuffle(shuffled)
        strategies.append(shuffled)
        
    for strat in strategies:
        if time.time() - start_time > deadline - 10:
            break
        res = greedy_select(strat)
        if len(res) > len(best_set):
            best_set = set(res)
            
    # 2. Local Search Improvement
    current_set = list(best_set)
    current_set_obj = set(current_set)
    
    improved = True
    while improved:
        improved = False
        
        # Pass 1: Add any possible vertex
        for a in candidates:
            if a not in current_set_obj:
                if not (conflict[a] & current_set_obj):
                    current_set.append(a)
                    current_set_obj.add(a)
                    improved = True
        if improved:
            continue # Restart loop as set changed
            
        # Pass 2: Swap/Exchange
        # Try removing one element and adding all possible new ones
        for i in range(len(current_set)):
            if time.time() - start_time > deadline - 5:
                return current_set
            
            rem = current_set[i]
            remaining = [x for x in current_set if x != rem]
            remaining_set = set(remaining)
            
            # Find all candidates that can be added to remaining_set
            addable = []
            for a in candidates:
                if a in remaining_set:
                    continue
                if a == rem:
                    continue
                if not (conflict[a] & remaining_set):
                    addable.append(a)
            
            # If we can add more than 1, size increases
            # If we can add 1, size stays same (might help later?)
            # If we add 0, size decreases (bad)
            
            if len(addable) > 0:
                new_set = remaining + addable
                new_set_obj = set(new_set)
                
                # We accept if size increases OR if size is same and we haven't tried this specific swap recently?
                # For now, only accept strict improvement to avoid cycles.
                # Actually, if len(addable) == 1, size is same. 
                # If len(addable) > 1, size increases.
                
                if len(new_set) > len(current_set):
                    current_set = new_set
                    current_set_obj = new_set_obj
                    improved = True
                    break # Restart loop with new set
                # Optional: Accept equal size swaps to escape local minima?
                # Risky without cycle detection. Let's stick to strict improvement for safety.
        
    return current_set
