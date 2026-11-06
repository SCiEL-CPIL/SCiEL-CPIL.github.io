import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    best_set = []
    best_size = 0
    
    n = 4096
    
    def try_greedy(candidates):
        S = []
        sums = set()
        
        for x in candidates:
            if time.time() > deadline:
                return S
            
            # Generate new sums
            new_sums = []
            
            new_sums.append(3*x)
            
            for y in S:
                new_sums.append(2*x + y)
                
            for y in S:
                new_sums.append(x + 2*y)
                
            n_s = len(S)
            for i in range(n_s):
                y = S[i]
                for j in range(i+1, n_s):
                    z = S[j]
                    new_sums.append(x + y + z)
            
            # Check for internal collisions
            seen = set()
            valid = True
            for s in new_sums:
                if s in seen:
                    valid = False
                    break
                seen.add(s)
            
            if not valid:
                continue

            # Check for collision with existing sums
            for s in new_sums:
                if s in sums:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Valid addition
            S.append(x)
            sums.update(new_sums)
            
            if len(S) >= 80:
                return S

        return S

    # 1. Deterministic increasing
    candidates = list(range(1, n + 1))
    S = try_greedy(candidates)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
        
    # 2. Deterministic decreasing
    candidates = list(range(n, 0, -1))
    S = try_greedy(candidates)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
        
    # 3. Random restarts
    attempts = 0
    while time.time() < deadline - 2:
        random.seed(attempts)
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        S = try_greedy(candidates)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
        attempts += 1
        
    return best_set
