import numpy as np
import time

def construct():
    N = 512
    best_size = 0
    best_set = set()
    start_time = time.time()
    max_time = 55  # Leave some buffer for 60s limit
    
    # Precompute indices for each position x
    # For x as 1st term: x, x+d, x+2d, x+3d. Need x+d, x+2d, x+3d in S.
    # d ranges from 1 to (N-1-x)//3
    # For x as 2nd term: x-d, x, x+d, x+2d. Need x-d, x+d, x+2d in S.
    # d ranges from 1 to min(x-1, (N-1-x)//2)
    # For x as 3rd term: x-2d, x-d, x, x+d. Need x-2d, x-d, x+d in S.
    # d ranges from 1 to min((x-1)//2, N-1-x)
    # For x as 4th term: x-3d, x-2d, x-d, x. Need x-3d, x-2d, x-d in S.
    # d ranges from 1 to (x-1)//3
    
    # We will store the indices for each x and each role
    # To save memory and time, we compute them on the fly or precompute.
    # Precomputing for all x is better.
    
    # idx_1st[x] = list of (x+d, x+2d, x+3d) for valid d
    # Actually, we just need the arrays for vectorization.
    
    # Let's precompute the index arrays for each x.
    # Since N is small, we can store lists of tuples or arrays.
    # Storing arrays allows direct indexing S[indices].
    
    idx_1st = [None] * (N + 1)
    idx_2nd = [None] * (N + 1)
    idx_3rd = [None] * (N + 1)
    idx_4th = [None] * (N + 1)
    
    for x in range(1, N + 1):
        # 1st term
        max_d1 = (N - x) // 3
        if max_d1 > 0:
            d1 = np.arange(1, max_d1 + 1)
            idx_1st[x] = (x + d1, x + 2 * d1, x + 3 * d1)
        else:
            idx_1st[x] = None
            
        # 2nd term
        max_d2 = min(x - 1, (N - x) // 2)
        if max_d2 > 0:
            d2 = np.arange(1, max_d2 + 1)
            idx_2nd[x] = (x - d2, x + d2, x + 2 * d2)
        else:
            idx_2nd[x] = None
            
        # 3rd term
        max_d3 = min((x - 1) // 2, N - x)
        if max_d3 > 0:
            d3 = np.arange(1, max_d3 + 1)
            idx_3rd[x] = (x - 2 * d3, x - d3, x + d3)
        else:
            idx_3rd[x] = None
            
        # 4th term
        max_d4 = (x - 1) // 3
        if max_d4 > 0:
            d4 = np.arange(1, max_d4 + 1)
            idx_4th[x] = (x - 3 * d4, x - 2 * d4, x - d4)
        else:
            idx_4th[x] = None

    # Randomized Greedy Trials
    # Use numpy random for speed
    while time.time() - start_time < max_time:
        S = np.zeros(N + 1, dtype=bool)
        # Shuffle order
        order = np.random.permutation(1, N + 1)
        
        for x in order:
            valid = True
            
            # Check 1st term
            if idx_1st[x] is not None:
                i1, i2, i3 = idx_1st[x]
                if np.any(S[i1] & S[i2] & S[i3]):
                    valid = False
            
            if valid and idx_2nd[x] is not None:
                i1, i2, i3 = idx_2nd[x]
                if np.any(S[i1] & S[i2] & S[i3]):
                    valid = False
            
            if valid and idx_3rd[x] is not None:
                i1, i2, i3 = idx_3rd[x]
                if np.any(S[i1] & S[i2] & S[i3]):
                    valid = False
            
            if valid and idx_4th[x] is not None:
                i1, i2, i3 = idx_4th[x]
                if np.any(S[i1] & S[i2] & S[i3]):
                    valid = False
            
            if valid:
                S[x] = True
        
        size = np.sum(S)
        if size > best_size:
            best_size = size
            # Extract set
            best_set = set(np.where(S)[0].tolist())
            
    return sorted(best_set)
