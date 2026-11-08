import time
import math

def score_set(U):
    if not U or 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 0
    # Compute U+U and U-U
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    num = len(diffs) / len(sums)
    if num <= 1:
        return 1.0
    den = math.log(2 * max_u + 1)
    if den == 0:
        return 1.0
    return 1 + math.log(num) / den

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Initial best
    best_set = [0, 1, 3, 7, 13]
    best_score = score_set(best_set)
    
    # Local search: try to swap or add/remove elements
    # Focus on small sets (size 3 to 8) and small max elements (up to 50)
    
    # Pre-compute candidates for small max U
    # We can iterate through all subsets of {0..50} of size 3..8? Too many.
    # Instead, hill climbing.
    
    current = best_set[:]
    current_score = best_score
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try swapping each element with a new value
        for i in range(len(current)):
            old_val = current[i]
            # Try values in range 0..30
            for new_val in range(0, 31):
                if new_val == old_val:
                    continue
                # Ensure 0 is in set
                if old_val == 0 and new_val != 0:
                    continue
                if 0 not in current and new_val != 0:
                    continue
                
                # Create new set
                new_set = current[:]
                new_set[i] = new_val
                # Remove duplicates and sort
                new_set = sorted(set(new_set))
                
                if 0 not in new_set:
                    continue
                
                # Check size
                if len(new_set) < 3 or len(new_set) > 8:
                    continue
                
                s = score_set(new_set)
                if s > current_score + 1e-9:
                    current_score = s
                    current = new_set
                    improved = True
                    if s > best_score:
                        best_score = s
                        best_set = new_set[:]
        
        # Try adding an element
        if len(current) < 8:
            for new_val in range(0, 31):
                if new_val in current:
                    continue
                new_set = sorted(current + [new_val])
                if 0 not in new_set:
                    continue
                s = score_set(new_set)
                if s > current_score + 1e-9:
                    current_score = s
                    current = new_set
                    improved = True
                    if s > best_score:
                        best_score = s
                        best_set = new_set[:]
                        
        # Try removing an element (if size > 3)
        if len(current) > 3:
            for i in range(len(current)):
                if current[i] == 0:
                    continue
                new_set = current[:i] + current[i+1:]
                s = score_set(new_set)
                if s > current_score + 1e-9:
                    current_score = s
                    current = new_set
                    improved = True
                    if s > best_score:
                        best_score = s
                        best_set = new_set[:]

    return best_set
