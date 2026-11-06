import time
import random
import numpy as np

def construct():
    deadline = time.time() + 55.0
    best_set = []
    N = 4096
    MAX_SUM = 3 * N
    
    def try_add(x, S, used_sums):
        # Check 3*x
        s = 3 * x
        if used_sums[s]:
            return False
        
        # Check 2*x + y and x + 2*y for all y in S
        for y in S:
            s1 = 2 * x + y
            if used_sums[s1]:
                return False
            s2 = x + 2 * y
            if used_sums[s2]:
                return False
        
        n = len(S)
        for i in range(n):
            y = S[i]
            for j in range(i + 1, n):
                z = S[j]
                s3 = x + y + z
                if used_sums[s3]:
                    return False
                
        # If all checks pass, update used_sums and S
        used_sums[3 * x] = True
        for y in S:
            used_sums[2 * x + y] = True
            used_sums[x + 2 * y] = True
        
        for i in range(n):
            y = S[i]
            for j in range(i + 1, n):
                z = S[j]
                used_sums[x + y + z] = True
                
        S.append(x)
        return True

    def greedy_fill(S, used_sums, candidates):
        # Sort candidates to add smaller ones first? Or larger?
        # Smaller ones usually fit better in gaps?
        # Let's try increasing order.
        for x in candidates:
            if time.time() > deadline:
                break
            try_add(x, S, used_sums)
            
    # Strategy 1: Start with powers of 3
    S1 = []
    val = 1
    while val <= N:
        S1.append(val)
        val *= 3
        
    # We need to initialize used_sums for S1
    used_sums1 = np.zeros(MAX_SUM + 1, dtype=bool)
    # Manually add sums for S1
    # Since S1 is small, we can just use try_add logic or a helper
    # Let's create a helper to build used_sums from a set
    def build_used_sums(S):
        us = np.zeros(MAX_SUM + 1, dtype=bool)
        n = len(S)
        for i in range(n):
            a = S[i]
            us[3*a] = True
            for j in range(i, n):
                b = S[j]
                if i == j:
                    # 3a already handled? No, 3a is i==j==k.
                    # We need to handle i,j,k unordered.
                    # The loop above in try_add handles distinct i,j.
                    # For initialization, let's just iterate all i<=j<=k.
                    pass
                # This is getting complicated. Let's just use try_add to build it?
                # No, try_add modifies S.
                # Let's just compute all sums.
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    s = S[i] + S[j] + S[k]
                    us[s] = True
        return us

    used_sums1 = build_used_sums(S1)
    
    # Candidates are numbers not in S1
    S1_set = set(S1)
    candidates1 = [x for x in range(1, N + 1) if x not in S1_set]
    greedy_fill(S1, used_sums1, candidates1)
    
    if len(S1) > len(best_set):
        best_set = list(S1)

    # Strategy 2: Start with powers of 2
    S2 = []
    val = 1
    while val <= N:
        S2.append(val)
        val *= 2
        
    used_sums2 = build_used_sums(S2)
    S2_set = set(S2)
    candidates2 = [x for x in range(1, N + 1) if x not in S2_set]
    greedy_fill(S2, used_sums2, candidates2)
    
    if len(S2) > len(best_set):
        best_set = list(S2)

    # Strategy 3: Random starts
    for seed in range(50):
        if time.time() > deadline:
            break
        random.seed(seed)
        S3 = []
        # Pick 5 random large numbers
        large_candidates = random.sample(range(N//2, N + 1), 5)
        large_candidates.sort()
        for x in large_candidates:
            try_add(x, S3, np.zeros(MAX_SUM + 1, dtype=bool)) # This is wrong, need persistent used_sums
        
        # Let's redo Strategy 3 properly
        if time.time() > deadline:
            break
            
        S3 = []
        used_sums3 = np.zeros(MAX_SUM + 1, dtype=bool)
        # Pick a few random numbers to start
        start_size = random.randint(3, 8)
        start_nums = random.sample(range(1, N + 1), start_size)
        for x in sorted(start_nums):
            try_add(x, S3, used_sums3)
            
        S3_set = set(S3)
        candidates3 = [x for x in range(1, N + 1) if x not in S3_set]
        random.shuffle(candidates3)
        
        greedy_fill(S3, used_sums3, candidates3)
        
        if len(S3) > len(best_set):
            best_set = list(S3)

    return best_set
