from fractions import Fraction
import time

def construct():
    start_time = time.time()
    # Allow 85 seconds, leaving 5s for overhead
    deadline = start_time + 85.0
    
    best_set = []
    best_size = 0
    
    # Candidates sorted by reciprocal ascending (n descending)
    # This heuristic helps in finding larger sets by adding "small" elements first
    candidates = list(range(22, 0, -1))
    recips = {n: Fraction(1, n) for n in candidates}
    
    def backtrack(idx, current_set, current_sums):
        nonlocal best_set, best_size
        
        # Time check
        if time.time() > deadline:
            return
            
        # Update best
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
            
        # Pruning: if we can't beat best_size
        if len(current_set) + (len(candidates) - idx) <= best_size:
            return
            
        if idx >= len(candidates):
            return
            
        # Try including candidates[idx]
        n = candidates[idx]
        r = recips[n]
        
        # Check if adding r keeps the set dissociated
        # New sums are s + r for s in current_sums
        # Collision if (s + r) in current_sums for any s
        valid = True
        # Optimization: only check s such that s + r could be in current_sums
        # Since all sums are positive, and r > 0, s + r > s.
        # We can just iterate.
        for s in current_sums:
            if s + r in current_sums:
                valid = False
                break
        
        if valid:
            new_sums = set(current_sums)
            for s in current_sums:
                new_sums.add(s + r)
                
            current_set.append(n)
            backtrack(idx + 1, current_set, new_sums)
            current_set.pop()
            
        # Try excluding candidates[idx]
        backtrack(idx + 1, current_set, current_sums)
        
    backtrack(0, [], {Fraction(0)})
    
    # Fallback
    if not best_set:
        return [1]
        
    return best_set
