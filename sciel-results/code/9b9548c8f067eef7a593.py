import time
import random

def construct():
    deadline = time.time() + 85.0
    best_set = []
    
    def try_add(x, S, used_sums):
        new_sums = set()
        s = 3 * x
        if s in used_sums:
            return False
        new_sums.add(s)
        
        for y in S:
            s1 = 2 * x + y
            if s1 in used_sums or s1 in new_sums:
                return False
            new_sums.add(s1)
            
            s2 = x + 2 * y
            if s2 in used_sums or s2 in new_sums:
                return False
            new_sums.add(s2)
            
        n = len(S)
        for i in range(n):
            y = S[i]
            for j in range(i + 1, n):
                z = S[j]
                s3 = x + y + z
                if s3 in used_sums or s3 in new_sums:
                    return False
                new_sums.add(s3)
                
        used_sums.update(new_sums)
        S.append(x)
        return True

    def greedy_from_start(start_candidates):
        """Greedy: try to add each candidate in order, keep going until we can't add more."""
        S = []
        used_sums = set()
        
        # First try to add the starting candidates
        for g in start_candidates:
            try_add(g, S, used_sums)
        
        # Then try all remaining numbers in increasing order
        for x in range(1, 4097):
            if x in used_sums or x in S:
                continue
            # Check if x is already in S
            if x in S:
                continue
            try_add(x, S, used_sums)
            if time.time() > deadline:
                break
        
        return S

    # Strategy 1: Pure greedy from scratch
    S = greedy_from_start([])
    if len(S) > len(best_set):
        best_set = list(S)
    
    # Strategy 2: Greedy with different starting points
    start_strategies = [
        [1],
        [1, 2],
        [1, 3],
        [1, 2, 4],
        [1, 3, 5],
        [2, 3],
        [1, 4, 7],
        [1, 2, 5, 10],
        [1, 3, 6, 12],
        [1, 2, 4, 8, 16],
    ]
    
    for strat in start_strategies:
        if time.time() > deadline:
            break
        S = greedy_from_start(strat)
        if len(S) > len(best_set):
            best_set = list(S)
    
    # Strategy 3: Random restarts with greedy fill
    for seed in range(200):
        if time.time() > deadline:
            break
        random.seed(seed)
        
        S = []
        used_sums = set()
        
        # Random start with a few elements
        num_start = random.randint(1, 5)
        start_candidates = random.sample(range(1, 4097), num_start)
        start_candidates.sort()
        
        for g in start_candidates:
            try_add(g, S, used_sums)
        
        # Greedy fill from small to large
        for x in range(1, 4097):
            if x in S:
                continue
            try_add(x, S, used_sums)
            if time.time() > deadline:
                break
        
        if len(S) > len(best_set):
            best_set = list(S)
    
    # Strategy 4: Random order with backtracking-lite (try multiple candidates at each step)
    for seed in range(100):
        if time.time() > deadline:
            break
        random.seed(seed + 10000)
        
        S = []
        used_sums = set()
        
        # Build up to 10 elements randomly
        for _ in range(10):
            if time.time() > deadline:
                break
            # Try several random candidates
            found = False
            for _ in range(50):
                x = random.randint(1, 4096)
                if x not in S:
                    # Test if we can add x
                    new_sums = set()
                    s = 3 * x
                    if s in used_sums:
                        continue
                    new_sums.add(s)
                    
                    ok = True
                    for y in S:
                        s1 = 2 * x + y
                        if s1 in used_sums or s1 in new_sums:
                            ok = False
                            break
                        new_sums.add(s1)
                        
                        s2 = x + 2 * y
                        if s2 in used_sums or s2 in new_sums:
                            ok = False
                            break
                        new_sums.add(s2)
                    
                    if not ok:
                        continue
                    
                    n = len(S)
                    for i in range(n):
                        y = S[i]
                        for j in range(i + 1, n):
                            z = S[j]
                            s3 = x + y + z
                            if s3 in used_sums or s3 in new_sums:
                                ok = False
                                break
                            new_sums.add(s3)
                        if not ok:
                            break
                    
                    if ok:
                        used_sums.update(new_sums)
                        S.append(x)
                        found = True
                        break
            
            if not found:
                break
        
        # Greedy fill
        for x in range(1, 4097):
            if x in S:
                continue
            try_add(x, S, used_sums)
            if time.time() > deadline:
                break
        
        if len(S) > len(best_set):
            best_set = list(S)
    
    return best_set
