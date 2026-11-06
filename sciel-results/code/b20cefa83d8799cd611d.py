import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_size = 0
    best_set = []
    
    while time.time() < deadline:
        candidates = list(range(1, 1001))
        random.shuffle(candidates)
        
        S = []
        rep_counts = [0] * 2001
        
        for x in candidates:
            valid = True
            if rep_counts[2*x] >= 2:
                valid = False
            else:
                for y in S:
                    if y <= x and rep_counts[x+y] >= 2:
                        valid = False
                        break
            
            if valid:
                S.append(x)
                rep_counts[2*x] += 1
                for y in S:
                    if y <= x:
                        rep_counts[x+y] += 1
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
    return best_set
