import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n_max = 4096
    candidates = list(range(1, n_max + 1))
    
    best_set = []
    best_size = 0
    
    def try_add(S, sums_set, x):
        # Check for collisions
        # 1. 3x
        if 3 * x in sums_set:
            return False
        # 2. 2x + a
        for a in S:
            if 2 * x + a in sums_set:
                return False
        # 3. x + a + b
        len_S = len(S)
        for i in range(len_S):
            a = S[i]
            for j in range(i, len_S):
                b = S[j]
                if x + a + b in sums_set:
                    return False
        
        # No collisions, add to sums_set
        sums_set.add(3 * x)
        for a in S:
            sums_set.add(2 * x + a)
        for i in range(len_S):
            a = S[i]
            for j in range(i, len_S):
                b = S[j]
                sums_set.add(x + a + b)
                
        return True

    # Initial deterministic-ish run to ensure validity
    random.seed(0)
    random.shuffle(candidates)
    S = []
    sums_set = set()
    for x in candidates:
        if try_add(S, sums_set, x):
            S.append(x)
    best_set = S
    best_size = len(S)
    
    run_count = 1
    while time.time() < deadline:
        random.seed(run_count)
        random.shuffle(candidates)
        
        S = []
        sums_set = set()
        
        for x in candidates:
            if try_add(S, sums_set, x):
                S.append(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
        run_count += 1
        
    return best_set
