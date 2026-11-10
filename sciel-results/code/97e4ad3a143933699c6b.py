import random
import time

def construct():
    best_set = []
    best_size = 0
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Optional: Try a few deterministic heuristics first?
    # But random is usually better for diversity.
    
    while time.time() < deadline:
        candidates = list(range(1, 1001))
        random.shuffle(candidates)
        
        S = []
        sum_counts = {}
        
        for x in candidates:
            # Check if x can be added
            valid = True
            if sum_counts.get(2*x, 0) >= 2:
                valid = False
            else:
                for y in S:
                    if sum_counts.get(x + y, 0) >= 2:
                        valid = False
                        break
            
            if valid:
                S.append(x)
                # Update counts
                sum_counts[2*x] = sum_counts.get(2*x, 0) + 1
                for y in S:
                    s = x + y
                    sum_counts[s] = sum_counts.get(s, 0) + 1
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
    return best_set
