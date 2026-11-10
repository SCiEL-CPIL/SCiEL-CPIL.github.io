import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for safety
    
    best_set = []
    
    # Run multiple randomized greedy attempts
    # Each attempt shuffles the candidate set and greedily adds elements
    # if they don't violate the B2[2] constraint.
    
    while time.time() < deadline:
        candidates = list(range(1, 1001))
        random.shuffle(candidates)
        
        current_set = []
        counts = {}
        
        for x in candidates:
            valid = True
            
            # Check if adding x creates any sum with > 2 representations
            # The new sums introduced by x are:
            # 1. x + y for each y in current_set
            # 2. 2 * x
            
            # Check against existing counts
            for y in current_set:
                s = x + y
                if counts.get(s, 0) >= 2:
                    valid = False
                    break
            
            if valid:
                s2 = 2 * x
                if counts.get(s2, 0) >= 2:
                    valid = False
            
            if valid:
                # Update counts with the new sums
                for y in current_set:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
                counts[2 * x] = counts.get(2 * x, 0) + 1
                
                # Add x to the current set
                current_set.append(x)
        
        # Keep the best set found so far
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            
    return best_set
