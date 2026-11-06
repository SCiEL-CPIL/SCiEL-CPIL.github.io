import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_set = []
    best_size = 0
    
    # Pre-allocate list for candidates to avoid recreation overhead in tight loop?
    # list(range(1, 1001)) is fast enough.
    
    while time.time() < deadline:
        candidates = list(range(1, 1001))
        random.shuffle(candidates)
        
        S = []
        counts = [0] * 2001 # Sums range from 2 to 2000
        
        for x in candidates:
            can_add = True
            
            # Check 2x
            if counts[2 * x] >= 2:
                can_add = False
            
            if can_add:
                # Check x + y for all y in S
                # Optimization: if S is large, this loop is the bottleneck.
                # But |S| is small (~60).
                for y in S:
                    if counts[x + y] >= 2:
                        can_add = False
                        break
            
            if can_add:
                S.append(x)
                counts[2 * x] += 1
                for y in S:
                    if y != x:
                        counts[x + y] += 1
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
    return best_set
