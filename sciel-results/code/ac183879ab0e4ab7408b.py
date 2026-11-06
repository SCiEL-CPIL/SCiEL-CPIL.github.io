import random
import time

def construct():
    N = 512
    best_set = set()
    start_time = time.time()
    max_time = 55  # Leave some buffer for 60s limit
    
    # Precompute all 4-APs in [1, N] as tuples of 4 elements (0-indexed for bitset)
    # An AP is (a, a+d, a+2d, a+3d) where 1 <= a, a+3d <= N
    aps = []
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # For fast lookup: for each element x, store list of indices of APs where x appears
    # And for each AP, store its elements
    # We'll use a bitset approach: maintain a set S, and for each candidate x,
    # check if adding x creates any 4-AP.
    
    # Alternative: maintain for each pair (i, j) in S, which d values would complete an AP.
    # Actually, let's use a different approach:
    # For each x, the conflicting conditions are:
    # x is 1st: need x+d, x+2d, x+3d all in S
    # x is 2nd: need x-d, x+d, x+2d all in S
    # x is 3rd: need x-2d, x-d, x+d all in S
    # x is 4th: need x-3d, x-2d, x-d all in S
    
    # To speed this up, we can maintain:
    # For each pair (u, v) in S with u < v, d = v - u.
    # Then:
    # - If x is 1st term: x, x+d, x+2d, x+3d -> we need (x+d, x+2d) and (x+2d, x+3d) pairs? 
    #   Actually, we need all three: x+d, x+2d, x+3d in S.
    #   This is equivalent to: for each d, check if x+d, x+2d, x+3d are all in S.
    
    # Let's use numpy bitsets for faster membership testing.
    import numpy as np
    
    # Bitset of size N+1 (indices 1..N)
    # We'll use a Python set for S but also maintain auxiliary structures.
    
    # Let's try a different optimization:
    # Maintain a "conflict count" for each candidate x.
    # When we add element y to S, we update conflict counts for all x that would form a 4-AP with y.
    
    # For a fixed y in S, which x's are affected?
    # x is 1st: x, x+d, x+2d, x+3d with y being one of x+d, x+2d, x+3d
    #   If y = x+d: x = y-d, need y-2d? No, x+2d = y+d, x+3d = y+2d. So need y+d, y+2d in S.
    #   If y = x+2d: x = y-2d, need x+d = y-d, x+3d = y+d in S. So need y-d, y+d in S.
    #   If y = x+3d: x = y-3d, need x+d = y-2d, x+2d = y-d in S. So need y-2d, y-d in S.
    
    # This is getting complex. Let's just use the bitset approach with numpy for fast checks.
    
    # Actually, let's use a simpler optimization:
    # Use a boolean array for membership, and precompute the AP checks more efficiently.
    
    def greedy_trial(seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        S = [False] * (N + 1)
        s_set = set()
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        
        for x in candidates:
            valid = True
            
            # Check if x is the 1st term: x, x+d, x+2d, x+3d
            max_d = (N - x) // 3
            for d in range(1, max_d + 1):
                if S[x + d] and S[x + 2 * d] and S[x + 3 * d]:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Check if x is the 2nd term: x-d, x, x+d, x+2d
            max_d = min(x - 1, (N - x) // 2)
            for d in range(1, max_d + 1):
                if S[x - d] and S[x + d] and S[x + 2 * d]:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Check if x is the 3rd term: x-2d, x-d, x, x+d
            max_d = min((x - 1) // 2, N - x)
            for d in range(1, max_d + 1):
                if S[x - 2 * d] and S[x - d] and S[x + d]:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Check if x is the 4th term: x-3d, x-2d, x-d, x
            max_d = (x - 1) // 3
            for d in range(1, max_d + 1):
                if S[x - 3 * d] and S[x - 2 * d] and S[x - d]:
                    valid = False
                    break
            
            if valid:
                S[x] = True
                s_set.add(x)
        
        return s_set
    
    # Run multiple randomized greedy trials
    trial_count = 0
    while time.time() - start_time < max_time:
        S = greedy_trial(trial_count)
        trial_count += 1
        
        if len(S) > len(best_set):
            best_set = S
    
    return sorted(best_set)
