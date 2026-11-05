import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe within 120s
    
    fixed = {1, 2, 5, 11}
    best_set = list(fixed)
    best_size = len(fixed)
    
    # Pre-calculate initial sums
    init_sums = set()
    fixed_list = list(fixed)
    for i in range(len(fixed_list)):
        for j in range(i, len(fixed_list)):
            init_sums.add(fixed_list[i] + fixed_list[j])
            
    # Candidates to consider
    candidates = [x for x in range(1, 285) if x not in fixed]
    
    # Branch and Bound Backtracking
    # We sort candidates in increasing order to try to pack them densely
    # This helps the bound (284 - current_idx) to be tighter if we skip many?
    # Actually, if we iterate 1..284, the bound is (284 - i).
    
    def check_add(x, current_sums):
        # Check if x can be added to the current set
        # We need to check sums with all elements in the current set
        # But we don't have the current set elements in the recursive call easily?
        # We need to pass the current set or maintain it.
        pass

    # Let's implement a cleaner backtracking
    current_set = list(fixed)
    current_sums = set(init_sums)
    
    def backtrack(idx, current_set, current_sums, current_size):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
            
        # Pruning
        # Max possible size if we add all remaining candidates
        remaining = len(candidates) - idx
        if current_size + remaining <= best_size:
            return
            
        # Base case: no more candidates
        if idx == len(candidates):
            if current_size > best_size:
                best_size = current_size
                best_set = list(current_set)
            return
            
        x = candidates[idx]
        
        # Option 1: Try to include x
        # Check if x can be added
        valid = True
        new_sums = set()
        for y in current_set:
            s = x + y
            if s in current_sums:
                valid = False
                break
            new_sums.add(s)
            
        if valid:
            # Add x
            current_set.append(x)
            current_sums.update(new_sums)
            # Note: self-sum x+x is also a pairwise sum if a<=b allows a==b?
            # Task says: "pairwise sums a + b (a <= b)". Usually this implies distinct elements or multiset?
            # "distinct integers... containing...".
            # Standard Sidon set definition: all sums a_i + a_j for i <= j are distinct.
            # So x + x must also be checked.
            if x + x in current_sums:
                 valid = False
            else:
                 current_sums.add(x + x)
                 
        if valid:
            backtrack(idx + 1, current_set, current_sums, current_size + 1)
            
        # Undo
        if valid:
            current_set.pop()
            for s in new_sums:
                current_sums.remove(s)
            if x + x in current_sums: # It was added
                current_sums.remove(x + x)
                
        # Option 2: Skip x
        backtrack(idx + 1, current_set, current_sums, current_size)

    # Initial call
    # To improve performance, we can sort candidates. 
    # Often, smaller numbers are "more constraining" or allow more packing?
    # Let's just use natural order 1..284.
    
    # Optimization: If we find a set of size 20, we can stop.
    # The known max is 20.
    
    backtrack(0, current_set, current_sums, len(current_set))
    
    return best_set
