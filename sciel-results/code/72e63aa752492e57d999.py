import numpy as np

def construct():
    n = 50
    
    # Backtracking to find max Sidon set in {0, ..., n-1}
    best_set = []
    
    def backtrack(start, current_set, used_sums):
        nonlocal best_set
        # Pruning: if we can't beat the current best size, stop
        # Max possible size is len(current_set) + (n - start)
        if len(current_set) + (n - start) <= len(best_set):
            return
        
        # Update best if current is better
        if len(current_set) > len(best_set):
            best_set = current_set.copy()
            
        # Try adding each index from 'start' to n-1
        for i in range(start, n):
            # Check if i can be added
            valid = True
            new_sums = []
            
            # Check sums with existing elements
            for j in current_set:
                s = i + j
                if s in used_sums:
                    valid = False
                    break
                new_sums.append(s)
            
            if valid:
                # Check self-sum
                s_ii = i + i
                if s_ii in used_sums:
                    valid = False
                else:
                    new_sums.append(s_ii)
            
            if valid:
                # Recurse
                used_sums.update(new_sums)
                current_set.append(i)
                
                backtrack(i + 1, current_set, used_sums)
                
                current_set.pop()
                used_sums.difference_update(new_sums)
                
                # Optional: If we found a large set, we can stop early?
                # Theoretical max is ~sqrt(n). For n=50, max is 7 or 8.
                # If we find 8, we can stop.
                if len(best_set) >= 8:
                    return

    backtrack(0, [], set())
    
    # Construct h
    k = len(best_set)
    if k == 0:
        # Fallback
        h = np.zeros(n)
        h[0] = 1.0
        return list(h)
        
    h = np.zeros(n)
    for idx in best_set:
        h[idx] = 1.0 / k
        
    return list(h)
