import random
import time

def construct():
    N = 512
    best_set = set()
    start_time = time.time()
    max_time = 55  # Leave some buffer for 60s limit
    
    # Run multiple randomized greedy trials
    while time.time() - start_time < max_time:
        S = set()
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            valid = True
            
            # Check if x is the 1st term: x, x+d, x+2d, x+3d
            max_d = (N - x) // 3
            for d in range(1, max_d + 1):
                if (x + d) in S and (x + 2 * d) in S and (x + 3 * d) in S:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Check if x is the 2nd term: x-d, x, x+d, x+2d
            max_d = min(x - 1, (N - x) // 2)
            for d in range(1, max_d + 1):
                if (x - d) in S and (x + d) in S and (x + 2 * d) in S:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Check if x is the 3rd term: x-2d, x-d, x, x+d
            max_d = min((x - 1) // 2, N - x)
            for d in range(1, max_d + 1):
                if (x - 2 * d) in S and (x - d) in S and (x + d) in S:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Check if x is the 4th term: x-3d, x-2d, x-d, x
            max_d = (x - 1) // 3
            for d in range(1, max_d + 1):
                if (x - 3 * d) in S and (x - 2 * d) in S and (x - d) in S:
                    valid = False
                    break
            
            if valid:
                S.add(x)
        
        if len(S) > len(best_set):
            best_set = S
            # If we hit a very high score, we might stop early, but let's just run until time limit
            # Typical max size for N=512 is around 150-200? 
            # Actually, r_4(512) is likely around 150-180.
            
    return sorted(best_set)
